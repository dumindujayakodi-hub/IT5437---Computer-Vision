import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#  Load Data

D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)

# Part (a): first line data only
X1 = D[:, 0]
Y1 = D[:, 3]

# Part (b): all 300 points (3 columns x 100 rows each)
X_all = D[:, :3].flatten()
Y_all = D[:, 3:].flatten()

print(f"Total points loaded: {len(X_all)}")
print(f"Line-1 points      : {len(X1)}\n")

#  Helper Functions
def fit_tls(x, y):
    xm, ym = x.mean(), y.mean()
    M = np.column_stack([x - xm, y - ym])      # centred data matrix
    _, _, Vt = np.linalg.svd(M)                 # singular value decomposition
    a, b = Vt[-1]                               # last row = smallest singular value
    c = -(a * xm + b * ym)                      # offset from centroid
    return a, b, c


def ransac_line(x, y, n_iter=2000, thresh=1.4, min_inliers=30, seed=42):
    rng = np.random.default_rng(seed)
    N = len(x)
    best_inliers = np.zeros(N, dtype=bool)
    best_count   = 0

    for _ in range(n_iter):
        
        idx = rng.choice(N, 2, replace=False)
        x1s, y1s = x[idx[0]], y[idx[0]]
        x2s, y2s = x[idx[1]], y[idx[1]]

        
        a = y2s - y1s
        b = -(x2s - x1s)
        c = (x2s - x1s) * y1s - (y2s - y1s) * x1s

       
        dists   = np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
        inliers = dists < thresh
        count   = np.sum(inliers)

  
        if count > best_count and count >= min_inliers:
            best_count   = count
            best_inliers = inliers


    a_r, b_r, c_r = fit_tls(x[best_inliers], y[best_inliers])
    return a_r, b_r, c_r, best_inliers

#  Part (a) — TLS on Line-1 data only

print("PART (a) — TLS on first line data (x1, y1)")


a1, b1, c1 = fit_tls(X1, Y1)

print(f"  Line equation : {a1:.6f}*x + {b1:.6f}*y + {c1:.6f} = 0")
print(f"  Normal vector : a = {a1:.6f},  b = {b1:.6f}")
print(f"  Offset        : c = {c1:.6f}")
print(f"  Slope form    : y = {-a1/b1:.4f}*x + {-c1/b1:.4f}\n")

#  Part (b) — RANSAC for 3 lines from all 300 points

print("PART (b) — RANSAC on all 300 points -> 3 lines")


remaining_mask = np.ones(len(X_all), dtype=bool)
lines     = []
all_masks = []

for i in range(3):
    rx = X_all[remaining_mask]
    ry = Y_all[remaining_mask]

    a, b, c, inliers_local = ransac_line(rx, ry, seed=42 + i)

    global_indices = np.where(remaining_mask)[0]
    gm = np.zeros(len(X_all), dtype=bool)
    gm[global_indices[inliers_local]] = True

    remaining_mask[gm] = False

    lines.append((a, b, c))
    all_masks.append(gm)

    print(f"\n  Line {i+1}:")
    print(f"    Equation    : {a:.6f}*x + {b:.6f}*y + {c:.6f} = 0")
    print(f"    Slope form  : y = {-a/b:.4f}*x + {-c/b:.4f}")
    print(f"    Inlier count: {gm.sum()}")

print(f"\n  Unassigned points: {remaining_mask.sum()}\n")


#  Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("IT5437 A2 — Q1: Line Fitting (TLS & RANSAC)",
             fontsize=14, fontweight='bold')


ax = axes[0]
ax.scatter(X1, Y1, color='steelblue', s=25, zorder=3, label='Line-1 data')
xr = np.linspace(X1.min() - 0.5, X1.max() + 0.5, 200)
ax.plot(xr, (-a1 * xr - c1) / b1, 'r-', lw=2.5,
        label=f'TLS: y={-a1/b1:.3f}x+{-c1/b1:.3f}')
ax.set_title("(a) TLS Fit — Line-1 Data Only", fontsize=11)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_aspect('equal')


ax = axes[1]
colours = ['#e74c3c', '#27ae60', '#2980b9']
handles = []

for i, (a, b, c) in enumerate(lines):
    m = all_masks[i]
    ax.scatter(X_all[m], Y_all[m], color=colours[i], s=20, alpha=0.75, zorder=3)
    xr = np.linspace(X_all[m].min() - 0.5, X_all[m].max() + 0.5, 200)
    ax.plot(xr, (-a * xr - c) / b, color=colours[i], lw=2.5)
    handles.append(mpatches.Patch(color=colours[i],
        label=f'L{i+1}: y={-a/b:.2f}x+{-c/b:.2f} ({m.sum()} pts)'))

if remaining_mask.any():
    ax.scatter(X_all[remaining_mask], Y_all[remaining_mask],
               color='gray', s=10, alpha=0.35)
    handles.append(mpatches.Patch(color='gray',
        label=f'Unassigned ({remaining_mask.sum()} pts)'))

ax.set_title("(b) RANSAC — 3 Lines from All 300 Points", fontsize=11)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(handles=handles, fontsize=8)
ax.grid(True, alpha=0.3); ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("q1_line_fitting.png", dpi=150, bbox_inches='tight')
plt.show()
