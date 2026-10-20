"""Animate the v3 cradle: one corner screw locking the phone in X and Z, then the roll adjustment.

Reads the exported STLs (world coordinates shared by every part), checks the corner
screw and roll adjust screw travel, drives the corner jaw along its 45 deg screw axis,
then rolls the cradle about its pivot bolt. Writes a GIF and an MP4.
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

# --- geometry from the CAD (mm), XZ plane unless noted --------------------
LENS_END_X = -82.0                    # end stop face
D45 = np.array([1.0, 1.0]) / np.sqrt(2)   # corner screw axis direction (X, Z): drives the jaw along -D45
P45 = np.array([1.0, -1.0]) / np.sqrt(2)  # across the screw axis
Q0 = np.array([73.5, 79.5])           # a point on the corner screw axis
SCREW_Y = -6.75                       # corner screw axis Y
BLOCK_IN, BLOCK_OUT = 22.0, 36.0      # screw block faces, along the axis from Q0
JAW_CORNER0 = np.array([68.0, 72.0])  # phone corner the jaw is modelled at (150 x 72 phone)
GROOVE0 = np.array([75.5, 79.5])      # groove centre on the jaw's 45 deg back face, as modelled
GROOVE_D, GROOVE_HALF = 2.0, 22.0     # groove depth and half length
SCREW_L, SCREW_R = 60.0, 3.0          # M6x60 corner thumbscrew
PIVOT = np.array([-15.0, -13.0])      # roll pivot bolt axis (along Y)
SEAT = np.array([60.0, -4.5])         # roll adjust screw seat under the lip (at 0 deg)
BASE_BOTTOM = -34.0                   # roll base underside: the M5 adjust screw enters here
ADJ_L = 40.0                          # M5x40 adjust screw
ROLL_MIN, ROLL_MAX = -6.5, 7.6        # deg: mount block hits the base / seat reaches the boss


def along(p):
    return float((np.asarray(p) - Q0) @ D45)


def check(phone_len, phone_w):
    corner = np.array([LENS_END_X + phone_len, phone_w])
    shift = corner - JAW_CORNER0                      # the jaw seats on the phone's corner
    tip = along(GROOVE0 + shift) - GROOVE_D            # screw tip, along the axis
    side = abs(float((GROOVE0 + shift - Q0) @ P45))    # tip offset along the groove
    shank = BLOCK_OUT - tip                            # shank needed past the head
    return dict(
        phone=f"{phone_len:.0f} x {phone_w:.0f}",
        shank_mm=round(shank, 1), groove_offset_mm=round(side, 1),
        jaw_to_block_mm=round(BLOCK_IN - (tip + GROOVE_D), 1),
        ok=shank <= SCREW_L and side <= GROOVE_HALF - SCREW_R and tip + GROOVE_D < BLOCK_IN,
    )


def roll_point(p, deg):
    a = np.radians(deg)
    r = np.asarray(p, float) - PIVOT
    return PIVOT + [r[0] * np.cos(a) - r[1] * np.sin(a), r[0] * np.sin(a) + r[1] * np.cos(a)]


def adjust_reach(deg):
    return roll_point(SEAT, deg)[1] - BASE_BOTTOM      # adjust screw length from the base underside to the seat


rows = [check(L, W) for L in (130, 150, 166) for W in (64, 72, 80)]
for r in rows:
    print(r)
assert all(r["ok"] for r in rows), "corner screw travel check failed"
reach = [adjust_reach(d) for d in (ROLL_MIN, 0.0, ROLL_MAX)]
print("adjust screw reach at", (ROLL_MIN, 0, ROLL_MAX), "deg:", [round(x, 1) for x in reach])
assert max(reach) <= ADJ_L and min(reach) > 34.0 - 16.0, "roll adjust screw check failed"


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
JAW_OPEN = 8.0                     # jaw starts backed off this far along the screw axis
ROLLS = [0.0, 4.0, -4.0, 0.0]      # roll demo, deg

N_IN, N_LOCK, N_ROLL, N_HOLD = 20, 30, 45, 12
N = N_IN + N_LOCK + N_ROLL + N_HOLD
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


cradle, jaw, base = (subdivide(load(n)) for n in ("cradle", "corner_jaw", "roll_base"))
mast = load("tripod_mast")
mast = subdivide(mast[mast[:, :, 2].min(1) > -110])


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def rolled(tris, deg):
    out = tris.copy()
    xz = out[..., [0, 2]].reshape(-1, 2)
    out[..., [0, 2]] = np.array([roll_point(p, deg) for p in xz]).reshape(out.shape[:-1] + (2,))
    return out


def xyz(p2, y):
    return [p2[0], y, p2[1]]


def frame(i):
    ax.cla()
    s_in = ease(i / N_IN)
    s_lock = ease((i - N_IN) / N_LOCK)
    k = np.clip((i - N_IN - N_LOCK) / N_ROLL * (len(ROLLS) - 1), 0, len(ROLLS) - 1 - 1e-9)
    j = int(k)
    deg = ROLLS[j] + (ROLLS[j + 1] - ROLLS[j]) * ease(k - j)

    dz_phone = (1 - s_in) * 60.0
    back = (1 - s_lock) * JAW_OPEN
    dj = D45 * back                                    # jaw offset in XZ

    moving = []                                        # parts that roll with the cradle
    moving.append((cradle, (0.93, 0.93, 0.90)))
    moving.append((jaw + [dj[0], 0, dj[1]], (0.95, 0.55, 0.25)))
    moving.append((subdivide(box(LENS_END_X, LENS_END_X + PL, -PT, 0.0, dz_phone, PW + dz_phone)), (0.15, 0.15, 0.18)))
    moving.append((cylinder([LENS_END_X + 35, 0.0, PW * 0.7 + dz_phone], [LENS_END_X + 35, 2.5, PW * 0.7 + dz_phone], 6), (0.05, 0.05, 0.05)))
    tip = along(GROOVE0) - GROOVE_D + back
    p_tip, p_head = Q0 + D45 * tip, Q0 + D45 * (tip + SCREW_L)
    moving.append((cylinder(xyz(p_tip, SCREW_Y), xyz(p_head, SCREW_Y), SCREW_R), (0.7, 0.7, 0.72)))
    moving.append((cylinder(xyz(p_head, SCREW_Y), xyz(p_head + D45 * 6, SCREW_Y), 8.0, 10), (0.85, 0.2, 0.2)))

    parts = [(rolled(t, deg), c) for t, c in moving]
    parts.append((base, (0.35, 0.62, 0.90)))
    parts.append((mast, (0.55, 0.55, 0.58)))
    # pivot bolt along Y, and the M5 roll adjust screw from below
    parts.append((cylinder([PIVOT[0], -29, PIVOT[1]], [PIVOT[0], 29, PIVOT[1]], 3.0), (0.7, 0.7, 0.72)))
    parts.append((cylinder([PIVOT[0], -33, PIVOT[1]], [PIVOT[0], -27, PIVOT[1]], 5.5, 6), (0.3, 0.3, 0.32)))
    seat_z = roll_point(SEAT, deg)[1]
    parts.append((cylinder([SEAT[0], 0, seat_z], [SEAT[0], 0, seat_z - ADJ_L], 2.5), (0.7, 0.7, 0.72)))
    parts.append((cylinder([SEAT[0], 0, seat_z - ADJ_L], [SEAT[0], 0, seat_z - ADJ_L - 6], 7.0, 10), (0.2, 0.65, 0.3)))

    tris = np.concatenate([t for t, _ in parts])
    cols = np.concatenate([shade(t, rgb) for t, rgb in parts])
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors="none", linewidths=0))

    ax.set_xlim(-110, 140)
    ax.set_ylim(-125, 125)
    ax.set_zlim(-95, 155)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-62)
    ax.dist = 6
    ax.set_axis_off()
    if i < N_IN:
        msg = "1. Phone drops in: lens end against the end stop, back on the rails"
    elif i < N_IN + N_LOCK:
        msg = "2. One M6 corner screw at 45 deg seats the jaw: locks X (end stop) and Z (bottom lip)"
    elif i < N_IN + N_LOCK + N_ROLL:
        msg = f"3. M5 adjust screw rolls the cradle about the pivot bolt: {deg:+.1f} deg"
    else:
        msg = "Locked and level: tighten the pivot bolt to hold the roll"
    ax.set_title(msg, fontsize=10)
    return []


anim = FuncAnimation(fig, frame, frames=N, interval=80)
anim.save(f"{OUT}.gif", writer=PillowWriter(fps=12))
anim.save(f"{OUT}.mp4", writer=FFMpegWriter(fps=12, bitrate=1800))
print("wrote", OUT)
