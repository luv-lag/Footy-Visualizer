"""evaluate.py -- score a video-pipeline run against SoccerNet GSR ground truth.

    # oracle detections (ground-truth boxes) -> isolates calibration/tracking
    python eval/gsr/evaluate.py oracle-cache <seq_dir> <out_dir>
    # after run_video.py has written <out_dir>/tracks.csv:
    python eval/gsr/evaluate.py score <seq_dir> <out_dir> [--viz viz.json]

Ground truth: ``Labels-GameState.json`` gives every person's foot position
in centre-origin pitch metres (``bbox_pitch.*_bottom_middle``).  Our
pipeline outputs corner-origin metres in a frame that is canonical per
camera side; which pitch end / side convention the labels use is not
knowable from geometry, so the scorer picks, once per clip, the one of the
four axis-sign conventions that best agrees (reported as ``convention``).

Metrics (people only: players, goalkeepers, referees):
  * calibrated_frames -- fraction of frames with a camera.
  * loc_err_m        -- median / p90 distance of matched (pred, gt) pairs.
  * recall/precision -- per-frame Hungarian matching with a 5 m gate.
  * idf1, id_switches -- identity consistency over the clip.
  * team_acc         -- team agreement (best label permutation).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

PEOPLE = {"player", "goalkeeper", "referee"}
GATE_M = 5.0
L, W = 105.0, 68.0


def load_gt(seq_dir: str) -> tuple[dict, dict]:
    """Return ({frame: [(gt_track, role, team, x_c, y_c)]}, info)."""
    with open(os.path.join(seq_dir, "Labels-GameState.json")) as f:
        lab = json.load(f)
    frame_of = {im["image_id"]: int(im["file_name"].split(".")[0]) - 1 for im in lab["images"]}
    gt: dict[int, list] = defaultdict(list)
    boxes: dict[int, list] = defaultdict(list)
    for a in lab["annotations"]:
        if a.get("supercategory") != "object":
            continue
        role = a["attributes"]["role"]
        f = frame_of[a["image_id"]]
        b = a["bbox_image"]
        boxes[f].append((role, b))
        if role in PEOPLE and a.get("bbox_pitch"):
            p = a["bbox_pitch"]
            gt[f].append((a["track_id"], role, a["attributes"].get("team"),
                          p["x_bottom_middle"], p["y_bottom_middle"]))
    return {"gt": gt, "boxes": boxes, "n_frames": len(lab["images"])}, lab["info"]


def write_oracle_cache(seq_dir: str, out_dir: str) -> None:
    """Ground-truth boxes as a detection cache (``--detector none``)."""
    data, _ = load_gt(seq_dir)
    cache = {}
    for f, items in data["boxes"].items():
        dets = []
        for role, b in items:
            if role not in PEOPLE | {"ball"}:
                continue
            cx, cy, w, h = b["x_center"], b["y_center"], b["w"], b["h"]
            dets.append({"class_name": role, "confidence": 1.0, "bbox_center": [cx, cy],
                         "bbox_size": [w, h], "foot_point": [cx, cy + h / 2]})
        cache[str(f)] = dets
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "detections.json"), "w") as fh:
        json.dump(cache, fh)


def load_pred(out_dir: str) -> tuple[dict, dict]:
    pred: dict[int, list] = defaultdict(list)
    with open(os.path.join(out_dir, "tracks.csv")) as f:
        for r in csv.DictReader(f):
            if r["cls"] == "ball" or "speed_outlier" in r["flags"] or not r["x_m"]:
                continue
            pred[int(r["frame"])].append((int(r["track_id"]), int(r["team"]),
                                          float(r["x_m"]), float(r["y_m"])))
    cam = {}
    with open(os.path.join(out_dir, "camera.csv")) as f:
        for r in csv.DictReader(f):
            cam[int(r["frame"])] = r["source"]
    return pred, cam


def to_center(x: float, y: float, sx: int, sy: int) -> tuple[float, float]:
    """Corner-origin prediction -> GT centre frame under sign convention (sx, sy)."""
    return sx * (x - L / 2), sy * (W / 2 - y)


def _match(P: np.ndarray, G: np.ndarray):
    if len(P) == 0 or len(G) == 0:
        return [], np.zeros((len(P), len(G)))
    D = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    r, c = linear_sum_assignment(np.where(D > GATE_M, 1e6, D))
    return [(i, j) for i, j in zip(r, c) if D[i, j] <= GATE_M], D


def score(seq_dir: str, out_dir: str, viz_path: str | None = None, viz_every: int = 2) -> dict:
    data, info = load_gt(seq_dir)
    gt, n_frames = data["gt"], data["n_frames"]
    pred, cam = load_pred(out_dir)

    def run(sx, sy, frames):
        per = {}
        for f in frames:
            P = np.array([to_center(x, y, sx, sy) for _, _, x, y in pred.get(f, [])]).reshape(-1, 2)
            G = np.array([(x, y) for *_, x, y in gt.get(f, [])]).reshape(-1, 2)
            per[f] = (_match(P, G)[0], P, G)
        return per

    # pick the axis convention on a subsample
    sample = list(range(0, n_frames, 10))
    conv = max(((sx, sy) for sx in (1, -1) for sy in (1, -1)),
               key=lambda c: sum(len(m) for m, _, _ in run(*c, sample).values()))
    per = run(*conv, range(n_frames))

    errs, n_gt, n_pred, n_match = [], 0, 0, 0
    co = Counter()  # (gt_track, pred_track) co-occurrence of matches
    last_pred_of: dict[int, int] = {}
    switches = 0
    team_pairs = Counter()
    frame_series = []
    for f in range(n_frames):
        matches, P, G = per[f]
        gts, prs = gt.get(f, []), pred.get(f, [])
        n_gt += len(gts)
        n_pred += len(prs)
        n_match += len(matches)
        fe = []
        for i, j in matches:
            e = float(np.linalg.norm(P[i] - G[j]))
            errs.append(e)
            fe.append(e)
            g_tid, p_tid = gts[j][0], prs[i][0]
            co[(g_tid, p_tid)] += 1
            if g_tid in last_pred_of and last_pred_of[g_tid] != p_tid:
                switches += 1
            last_pred_of[g_tid] = p_tid
            if prs[i][1] >= 0 and gts[j][2] in ("left", "right"):
                team_pairs[(prs[i][1], gts[j][2])] += 1
        frame_series.append({
            "f": f, "cam": cam.get(f, "none"),
            "err": round(float(np.median(fe)), 3) if fe else None,
            "recall": round(len(matches) / len(gts), 3) if gts else None,
        })

    # IDF1: global one-to-one gt<->pred identity assignment maximising matches
    g_ids = sorted({g for g, _ in co})
    p_ids = sorted({p for _, p in co})
    idtp = 0
    if g_ids and p_ids:
        M = np.zeros((len(g_ids), len(p_ids)))
        for (g, p), c in co.items():
            M[g_ids.index(g), p_ids.index(p)] = c
        r, c = linear_sum_assignment(-M)
        idtp = float(M[r, c].sum())
    a = team_pairs[(0, "left")] + team_pairs[(1, "right")]
    b = team_pairs[(0, "right")] + team_pairs[(1, "left")]
    team_total = a + b

    metrics = {
        "sequence": info.get("name"),
        "convention": {"sx": conv[0], "sy": conv[1]},
        "calibrated_frames": round(sum(1 for s in cam.values() if s != "lost") / max(1, len(cam)), 4),
        "loc_err_m": {"median": round(float(np.median(errs)), 3) if errs else None,
                      "p90": round(float(np.percentile(errs, 90)), 3) if errs else None},
        "recall": round(n_match / max(1, n_gt), 4),
        "precision": round(n_match / max(1, n_pred), 4),
        "idf1": round(2 * idtp / max(1, n_gt + n_pred), 4),
        "id_switches": switches,
        "gt_identities": len({t for v in gt.values() for t, *_ in v}),
        "pred_identities": len({t for v in pred.values() for t, *_ in v}),
        "team_acc": round(max(a, b) / team_total, 4) if team_total else None,
    }
    if viz_path:
        _write_viz(viz_path, metrics, per, gt, pred, frame_series, conv, viz_every,
                   team_flip=b > a)
    return metrics


def _write_viz(path, metrics, per, gt, pred, series, conv, every, team_flip) -> None:
    """Compact per-frame data for the interactive report (corner-origin metres,
    GT converted into the pipeline's frame so both share one pitch)."""
    sx, sy = conv

    def gt_to_corner(x, y):
        return round(sx * x + L / 2, 2), round(W / 2 - sy * y, 2)

    frames = []
    for f in range(0, len(series), every):
        matches, P, G = per[f]
        gts, prs = gt.get(f, []), pred.get(f, [])
        team_name = {"left": 0, "right": 1}
        frames.append({
            "f": f,
            "g": [[*gt_to_corner(x, y), t, {"player": "p", "goalkeeper": "g", "referee": "r"}[role],
                   (team_name.get(team, -1) ^ (1 if team_flip and team in team_name else 0))]
                  for t, role, team, x, y in gts],
            "p": [[round(x, 2), round(y, 2), tid, team] for tid, team, x, y in prs],
            "m": [[int(i), int(j)] for i, j in matches],
        })
    with open(path, "w") as fh:
        json.dump({"metrics": metrics, "series": series, "frames": frames, "every": every},
                  fh, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["oracle-cache", "score"])
    ap.add_argument("seq_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--viz")
    a = ap.parse_args()
    if a.cmd == "oracle-cache":
        write_oracle_cache(a.seq_dir, a.out_dir)
    else:
        print(json.dumps(score(a.seq_dir, a.out_dir, a.viz), indent=2))


if __name__ == "__main__":
    main()
