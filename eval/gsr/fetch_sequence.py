"""fetch_sequence.py -- pull single sequences out of the SoccerNet GSR zips.

The SN-GSR-2025 archives on Hugging Face are 5-11 GB each, but one 30 s
sequence is ~190 MB.  Hugging Face serves HTTP range requests, so this
reads the zip's central directory remotely and downloads only the member
files of the requested sequence(s).

    python eval/gsr/fetch_sequence.py --list                 # sequence names
    python eval/gsr/fetch_sequence.py --stats SNGS-021        # label stats only
    python eval/gsr/fetch_sequence.py SNGS-021 --out data/gsr # frames + labels
    python eval/gsr/fetch_sequence.py SNGS-021 --out data/gsr --mp4  # + clip.mp4

Dataset: https://huggingface.co/datasets/SoccerNet/SN-GSR-2025 (GPL-3.0).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

BASE = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/"


@dataclass
class Member:
    name: str
    method: int
    csize: int
    usize: int
    offset: int  # local header offset


class RemoteZip:
    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.members = self._central_directory()

    def _get(self, a: int, b: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except OSError:
                if attempt == 3:
                    raise
        raise AssertionError("unreachable")

    def _central_directory(self) -> dict[str, Member]:
        tail = self._get(max(0, self.size - 65558 - 20), self.size - 1)
        z64 = tail.rfind(b"PK\x06\x06")
        if z64 >= 0:
            cd_size, cd_off = struct.unpack("<QQ", tail[z64 + 40 : z64 + 56])
        else:
            eocd = tail.rfind(b"PK\x05\x06")
            cd_size, cd_off = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
        cd = self._get(cd_off, cd_off + cd_size - 1)
        out: dict[str, Member] = {}
        p = 0
        while cd[p : p + 4] == b"PK\x01\x02":
            method = struct.unpack("<H", cd[p + 10 : p + 12])[0]
            csize, usize = struct.unpack("<II", cd[p + 20 : p + 28])
            nl, el, cl = struct.unpack("<HHH", cd[p + 28 : p + 34])
            off = struct.unpack("<I", cd[p + 42 : p + 46])[0]
            name = cd[p + 46 : p + 46 + nl].decode()
            extra = cd[p + 46 + nl : p + 46 + nl + el]
            # zip64 extra field: values present only for fields set to 0xFFFFFFFF
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q : q + 4])
                if hid == 1:
                    vals = extra[q + 4 : q + 4 + hlen]
                    k = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[k : k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[k : k + 8])[0]; k += 8
                    if off == 0xFFFFFFFF:
                        off = struct.unpack("<Q", vals[k : k + 8])[0]; k += 8
                q += 4 + hlen
            out[name] = Member(name, method, csize, usize, off)
            p += 46 + nl + el + cl
        return out

    def read(self, m: Member) -> bytes:
        # local header is 30 bytes + name + extra (lengths may differ from the CD)
        blob = self._get(m.offset, m.offset + 30 + len(m.name) + 1024 + m.csize)
        nl, el = struct.unpack("<HH", blob[26:30])
        data = blob[30 + nl + el : 30 + nl + el + m.csize]
        if m.method == 0:
            return data
        if m.method == 8:
            return zlib.decompress(data, -15)
        raise ValueError(f"unsupported compression method {m.method} for {m.name}")

    def sequences(self) -> list[str]:
        return sorted({n.split("/")[0] for n in self.members if "/" in n})


def label_stats(labels: dict) -> dict:
    """Per-sequence summary used to pick main-camera-heavy clips."""
    anns = labels["annotations"]
    per_img: dict[str, int] = {}
    for a in anns:
        if a.get("supercategory") == "object" and a.get("bbox_pitch"):
            per_img[a["image_id"]] = per_img.get(a["image_id"], 0) + 1
    imgs = labels["images"]
    counts = [per_img.get(i["image_id"], 0) for i in imgs]
    return {
        "frames": len(imgs),
        "frames_with_people": sum(c > 0 for c in counts),
        "mean_people": round(sum(counts) / max(1, len(counts)), 1),
        "frames_lt_8_people": sum(c < 8 for c in counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sequences", nargs="*")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test", "challenge"])
    ap.add_argument("--out", help="output directory (one sub-directory per sequence)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stats", action="store_true", help="fetch labels only and print stats")
    ap.add_argument("--mp4", action="store_true", help="also write <seq>/clip.mp4")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    rz = RemoteZip(BASE + f"{a.split}.zip")
    if a.list:
        print("\n".join(rz.sequences()))
        return
    for seq in a.sequences:
        labels_name = f"{seq}/Labels-GameState.json"
        labels = json.loads(rz.read(rz.members[labels_name]))
        if a.stats:
            print(seq, json.dumps(label_stats(labels)))
            continue
        if not a.out:
            sys.exit("--out is required to download frames")
        dst = os.path.join(a.out, seq)
        os.makedirs(os.path.join(dst, "img1"), exist_ok=True)
        with open(os.path.join(dst, "Labels-GameState.json"), "w") as f:
            json.dump(labels, f)
        members = [m for n, m in rz.members.items() if n.startswith(f"{seq}/") and not n.endswith("/")
                   and n != labels_name]

        def fetch(m: Member) -> None:
            path = os.path.join(a.out, *m.name.split("/"))
            if os.path.isfile(path) and os.path.getsize(path) == m.usize:
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(rz.read(m))

        with ThreadPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(fetch, members), 1):
                if i % 100 == 0:
                    print(f"{seq}: {i}/{len(members)}", flush=True)
        if a.mp4:
            frames_to_mp4(os.path.join(dst, "img1"), os.path.join(dst, "clip.mp4"))
        print(f"{seq}: done -> {dst}")


def frames_to_mp4(img_dir: str, out: str, fps: float = 25.0) -> None:
    import cv2  # noqa: PLC0415

    names = sorted(n for n in os.listdir(img_dir) if n.endswith(".jpg"))
    first = cv2.imread(os.path.join(img_dir, names[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for n in names:
        vw.write(cv2.imread(os.path.join(img_dir, n)))
    vw.release()


if __name__ == "__main__":
    main()
