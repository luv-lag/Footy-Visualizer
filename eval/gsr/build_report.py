"""build_report.py -- bundle scored runs into the interactive accuracy report.

    python eval/gsr/build_report.py spec.json report.html

spec.json:
    {
      "clips": {"SNGS-048": {"label": "SNGS-048 · kick-off, hard shadow",
                             "runs": {"oracle": {"viz": "<run dir>",
                                                 "versions": [["original", "<run dir>"], ...]},
                                      "rfdetr": {...}}}},
      "thumbs": "<thumbs.json: {clip: [base64 jpeg per second]}>",
      "notes": [{"h": "...", "body": "<p>...</p>"}]
    }

Each run dir must hold ``metrics.json`` from ``evaluate.py score``; the
``viz`` dir (the version shown on the pitch) also needs ``viz.json`` (``--viz``).  Everything is inlined into one HTML file.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_LABELS = {"oracle": "Ground-truth boxes", "rfdetr": "RF-DETR (CPU)"}


def main() -> None:
    spec_path, out = sys.argv[1], sys.argv[2]
    with open(spec_path) as f:
        spec = json.load(f)
    clips = {}
    for ck, c in spec["clips"].items():
        runs = {}
        for rk, r in c["runs"].items():
            with open(os.path.join(r["viz"], "viz.json")) as f:
                viz = json.load(f)
            versions = []
            for label, d in r["versions"]:
                with open(os.path.join(d, "metrics.json")) as f:
                    versions.append({"label": label, "metrics": json.load(f)})
            runs[rk] = {"metrics": viz["metrics"], "versions": versions, "series": viz["series"],
                        "frames": viz["frames"], "every": viz["every"]}
        clips[ck] = {"label": c["label"], "runs": runs}
    data = {"clips": clips, "runLabels": RUN_LABELS, "notes": spec.get("notes", []),
            "generated": dt.date.today().isoformat()}
    with open(spec["thumbs"]) as f:
        thumbs = f.read()
    with open(os.path.join(HERE, "report_template.html")) as f:
        html = f.read()
    html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace("/*THUMBS*/null", thumbs)
    with open(out, "w") as f:
        f.write(html)
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
