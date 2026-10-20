"""Animate the v2 cradle's two screw grippers closing on a phone, and check the screw travel.

Reads the exported STLs (world coordinates shared by every part) and moves the
top clamp and end jaw along their screw axes. Writes a GIF and an MP4.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from stl import mesh

D = sys.argv[1]
OUT = sys.argv[2]

# --- geometry from the CAD (mm) -----------------------------------------
LENS_END_X = -82.0          # end stop face
JAW_FACE_X0 = 84.0          # end jaw face as modelled (166 mm phone)
JAW_T = 6.0
END_BLOCK_OUT = 103.0       # outer face of the end screw block
END_SCREW = dict(y=-4.0, z=25.0)
CLAMP_UNDER0 = 80.0         # clamp underside as modelled (80 mm phone)
CLAMP_TOP0 = 92.0
NUT_TOP = 109.0             # top of the top screw block
NUT_BOTTOM = 97.0
TOP_SCREW = dict(x=45.0, y=-4.0)
SLOT_Z = (70.0, 92.0)       # guide-slot range for the clamp's M5 guide bolts
SCREW_L_END, SCREW_L_TOP = 60.0, 40.0   # thumbscrew shank lengths


def check(phone_len, phone_w):
    end_x = LENS_END_X + phone_len
    end_need = END_BLOCK_OUT - (end_x + JAW_T)       # shank needed past the head
    top_need = NUT_TOP - (phone_w + 12.0)
    guide_z = phone_w + 6.0
    return dict(
        phone=f"{phone_len:.0f} x {phone_w:.0f}",
        end_screw_reach_mm=round(end_need, 1), end_ok=0 < end_need <= SCREW_L_END and end_x + JAW_T <= 91.0,
        top_screw_reach_mm=round(top_need, 1), top_ok=0 < top_need <= SCREW_L_TOP and phone_w + 12 <= NUT_BOTTOM,
        guide_bolt_z=guide_z, guide_ok=SLOT_Z[0] <= guide_z <= SLOT_Z[1],
    )


rows = [check(L, W) for L in (130, 150, 166) for W in (64, 72, 80)]
for r in rows:
    print(r)
assert all(r["end_ok"] and r["top_ok"] and r["guide_ok"] for r in rows), "screw travel check failed"


def load(name):
    m = mesh.Mesh.from_file(f"{D}/{name}.stl")
    return m.vectors.copy()


def box(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    q = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    tris = []
    for a, b, c, d in q:
        tris += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    return np.array(tris)


def cylinder(p0, p1, r, n=16):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax /= np.linalg.norm(ax)
    u = np.cross(ax, [0, 0, 1] if abs(ax[2]) < 0.9 else [1, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t)], 1)
    a = p0 + r * (ring[:, :1] * u + ring[:, 1:] * w)
    b = p1 + r * (ring[:, :1] * u + ring[:, 1:] * w)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [[a[i], a[j], b[j]], [a[i], b[j], b[i]], [p0, a[j], a[i]], [p1, b[i], b[j]]]
    return np.array(tris)



# demo phone: 150 x 72 x 8 mm
PL, PW, PT = 150.0, 72.0, 8.0
phone_end = LENS_END_X + PL
jaw_travel = phone_end - JAW_FACE_X0               # negative: jaw moves -X
clamp_travel = PW - CLAMP_UNDER0                    # negative: clamp moves down

N_IN, N_END, N_TOP, N_HOLD = 20, 30, 30, 15
N = N_IN + N_END + N_TOP + N_HOLD
LIGHT = np.array([0.3, -0.6, 0.75])
LIGHT /= np.linalg.norm(LIGHT)

fig = plt.figure(figsize=(8, 6), dpi=110)
fig.subplots_adjust(0, 0, 1, 0.94)
ax = fig.add_subplot(projection="3d")


def shade(tris, rgb):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    k = 0.45 + 0.55 * np.abs(n @ LIGHT)
    return np.clip(np.outer(k, rgb), 0, 1)


def subdivide(tris, max_edge=12.0):
    out = []
    stack = list(tris)
    while stack:
        t = stack.pop()
        e = [np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[1]), np.linalg.norm(t[0] - t[2])]
        if max(e) <= max_edge:
            out.append(t)
            continue
        k = int(np.argmax(e))
        a, b, c = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
        m = (a + b) / 2
        stack += [np.array([a, m, c]), np.array([m, b, c])]
    return np.array(out)


cradle, clamp, jaw = (subdivide(load(n)) for n in ("cradle", "top_clamp", "end_jaw"))


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def frame(i):
    ax.cla()
    s_in = ease(i / N_IN)
    s_end = ease((i - N_IN) / N_END)
    s_top = ease((i - N_IN - N_END) / N_TOP)

    dz_phone = (1 - s_in) * 60.0
    dx_jaw = s_end * jaw_travel
    dz_clamp = s_top * clamp_travel

    parts = []
    parts.append((cradle, (0.93, 0.93, 0.90)))
    parts.append((clamp + [0, 0, dz_clamp], (0.35, 0.62, 0.90)))
    parts.append((jaw + [dx_jaw, 0, 0], (0.95, 0.55, 0.25)))
    phone = subdivide(box(LENS_END_X, LENS_END_X + PL, -PT, 0.0, dz_phone, PW + dz_phone))
    parts.append((phone, (0.15, 0.15, 0.18)))
    lens = cylinder([LENS_END_X + 35, 0.0, PW * 0.7 + dz_phone], [LENS_END_X + 35, 2.5, PW * 0.7 + dz_phone], 6)
    parts.append((lens, (0.05, 0.05, 0.05)))

    # end thumbscrew: tip on the jaw's back face, knob outside the block
    tip_x = JAW_FACE_X0 + JAW_T + dx_jaw
    head_x = tip_x + SCREW_L_END
    parts.append((cylinder([tip_x, END_SCREW["y"], END_SCREW["z"]], [head_x, END_SCREW["y"], END_SCREW["z"]], 3.0), (0.7, 0.7, 0.72)))
    parts.append((cylinder([head_x, END_SCREW["y"], END_SCREW["z"]], [head_x + 6, END_SCREW["y"], END_SCREW["z"]], 8.0, 10), (0.85, 0.2, 0.2)))
    # top thumbscrew: tip on the clamp's top face
    tip_z = CLAMP_TOP0 + dz_clamp
    head_z = tip_z + SCREW_L_TOP
    parts.append((cylinder([TOP_SCREW["x"], TOP_SCREW["y"], tip_z], [TOP_SCREW["x"], TOP_SCREW["y"], head_z], 3.0), (0.7, 0.7, 0.72)))
    parts.append((cylinder([TOP_SCREW["x"], TOP_SCREW["y"], head_z], [TOP_SCREW["x"], TOP_SCREW["y"], head_z + 6], 8.0, 10), (0.85, 0.2, 0.2)))

    tris = np.concatenate([t for t, _ in parts])
    cols = np.concatenate([shade(t, rgb) for t, rgb in parts])
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors="none", linewidths=0))

    ax.set_xlim(-100, 150)
    ax.set_ylim(-125, 125)
    ax.set_zlim(-75, 175)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.dist = 6
    ax.set_axis_off()
    if i < N_IN:
        msg = "1. Phone drops in: lens end against the end stop, back on the rails"
    elif i < N_IN + N_END:
        msg = f"2. End thumbscrew (M6) drives the end jaw {abs(jaw_travel):.0f} mm onto the phone"
    elif i < N_IN + N_END + N_TOP:
        msg = f"3. Top thumbscrew (M6) drives the top clamp {abs(clamp_travel):.0f} mm down"
    else:
        msg = "Locked: phone held on all four edges, camera end and screen open"
    ax.set_title(msg, fontsize=10)
    return []


anim = FuncAnimation(fig, frame, frames=N, interval=80)
anim.save(f"{OUT}.gif", writer=PillowWriter(fps=12))
anim.save(f"{OUT}.mp4", writer=FFMpegWriter(fps=12, bitrate=1800))
print("wrote", OUT)
