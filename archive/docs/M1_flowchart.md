# M1 Pipeline Flowchart

```mermaid
flowchart TD
    A([Broadcast frame])
    B["Resize to 540x960<br/>SV_kp + SV_lines — HRNet-W48 heatmaps"]
    C["max-pool NMS<br/>sparse keypoints + line segments"]
    D["complete_keypoints + heuristic_voting<br/>camera model: K, R, t"]
    E{"n_kp &gt;= 4 and<br/>calibration valid?"}
    F([ABSTAIN — no positions emitted])
    G["H = K · r1 r2 t<br/>ground-plane homography"]
    H["Roboflow YOLOv8 → bounding boxes<br/>KMeans on jersey HSV → teams"]
    I["foot pixel u,v through H-inv<br/>→ PnLCalib world Xp, Yp"]
    J["axis swap + scale<br/>→ SynLoc metres x, y"]
    K([Top-down pitch plot])

    A --> B --> C --> D --> E
    E -- fail --> F
    E -- pass --> G
    G --> H --> I --> J --> K
```

PnLCalib recovers the full camera model from pitch-marking appearance; frames with fewer than 4 confident keypoints abstain rather than emit a degenerate homography; valid calibrations project each player's foot contact point through H-inv and a fixed coordinate conversion into metric pitch coordinates.
