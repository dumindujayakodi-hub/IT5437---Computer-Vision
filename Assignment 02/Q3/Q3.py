import numpy as np
import matplotlib.pyplot as plt
import cv2

im1 = cv2.imread("c1.jpg", cv2.IMREAD_REDUCED_COLOR_4)
im2 = cv2.imread("c2.jpg", cv2.IMREAD_REDUCED_COLOR_4)

if im1 is None or im2 is None:
    raise FileNotFoundError("c1.jpg or c2.jpg not found in current directory")

im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

h1, w1 = im1.shape[:2]
h2, w2 = im2.shape[:2]

print(f"c1 shape (reduced): {im1.shape}")
print(f"c2 shape (reduced): {im2.shape}\n")

#  PART (a) — Manual Correspondences -> Homography -> Warp
print("PART (a) — Manual Homography")
p1_manual = np.float32([
    [463, 202],   # 1. crystal oscillator area
    [337, 324],   # 2. main IC corner
    [374, 265],   # 3. IC edge
    [269, 215],   # 4. near USB region
    [420,  73],   # 5. USB connector corner
    [203, 412],   # 6. lower pin header
])

p2_manual = np.float32([
    [494, 221],
    [341, 306],
    [392, 259],
    [304, 182],
    [487,  86],
    [188, 355],
])

H_manual, mask_H = cv2.findHomography(p1_manual, p2_manual, 0)

print("  Homography matrix H (manual, 6 points):")
for row in H_manual:
    print(f"    [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.4f}]")
print()

# Warp c1 to match c2's perspective
warped_manual     = cv2.warpPerspective(im1, H_manual, (w2, h2))
warped_manual_rgb = cv2.cvtColor(warped_manual, cv2.COLOR_BGR2RGB)
print("  c1 warped to c2 perspective using manual H.\n")

#  PART (b) — Difference Image (Manual Homography)
print("PART (b) — Difference Image (Manual)")

# Pixel-wise absolute difference between warped c1 and c2
diff_manual = cv2.absdiff(warped_manual, im2)

# Amplify x3 so small differences become visible
diff_manual_amp     = cv2.convertScaleAbs(diff_manual, alpha=3.0)
diff_manual_rgb     = cv2.cvtColor(diff_manual,     cv2.COLOR_BGR2RGB)
diff_manual_amp_rgb = cv2.cvtColor(diff_manual_amp, cv2.COLOR_BGR2RGB)

mean_diff_manual = diff_manual.mean()
print(f"  Mean pixel difference (manual): {mean_diff_manual:.3f}")
print(f"  (Bright regions = misalignment)\n")

#  PART (c) — SIFT Keypoints, Descriptors, and Matching

print("PART (c) — SIFT Feature Matching")

sift = cv2.SIFT_create(nfeatures=1000)

kp1s, des1s = sift.detectAndCompute(im1, None)
kp2s, des2s = sift.detectAndCompute(im2, None)

print(f"  Keypoints in c1 : {len(kp1s)}")
print(f"  Keypoints in c2 : {len(kp2s)}")

FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches_raw = flann.knnMatch(des1s, des2s, k=2)

good_sift = [m for m, n in matches_raw if m.distance < 0.75 * n.distance]

print(f"  Good matches (Lowe ratio 0.75) : {len(good_sift)}\n")

# Draw top 40 matches for visualisation
match_img     = cv2.drawMatches(im1, kp1s, im2, kp2s, good_sift[:40], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)


#  PART (d) — SIFT Homography -> Warp -> Difference -> Comparison


print("PART (d) — SIFT-based Homography and Comparison")

src_pts = np.float32([kp1s[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2s[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)

H_sift, mask_sift = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
inliers_sift = int(mask_sift.sum())

print(f"  RANSAC inliers : {inliers_sift} / {len(good_sift)}")
print(f"  ({inliers_sift/len(good_sift)*100:.1f}% inlier ratio)")
print("\n  Homography matrix H (SIFT):")
for row in H_sift:
    print(f"    [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.4f}]")
print()


warped_sift     = cv2.warpPerspective(im1, H_sift, (w2, h2))
warped_sift_rgb = cv2.cvtColor(warped_sift, cv2.COLOR_BGR2RGB)

diff_sift         = cv2.absdiff(warped_sift, im2)
diff_sift_amp     = cv2.convertScaleAbs(diff_sift, alpha=3.0)
diff_sift_amp_rgb = cv2.cvtColor(diff_sift_amp, cv2.COLOR_BGR2RGB)

mean_diff_sift = diff_sift.mean()
print(f"  Mean pixel difference (SIFT)   : {mean_diff_sift:.3f}")
print(f"  Mean pixel difference (manual) : {mean_diff_manual:.3f}")
winner = 'SIFT' if mean_diff_sift < mean_diff_manual else 'Manual'
print(f"  Winner (lower = better)        : {winner}\n")


fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle("IT5437 A2 — Q3 Parts (a) & (b): Manual Homography",
              fontsize=14, fontweight='bold')

im1_pts = im1_rgb.copy()
im2_pts = im2_rgb.copy()
for i, (pt1, pt2) in enumerate(zip(p1_manual.astype(int), p2_manual.astype(int))):
    cv2.circle(im1_pts, tuple(pt1), 8, (255, 0, 0), -1)
    cv2.putText(im1_pts, str(i+1), tuple(pt1 + np.array([6, -6])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.circle(im2_pts, tuple(pt2), 8, (0, 255, 0), -1)
    cv2.putText(im2_pts, str(i+1), tuple(pt2 + np.array([6, -6])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

axes1[0, 0].imshow(im1_rgb);           axes1[0, 0].set_title("c1 — Original");             axes1[0, 0].axis('off')
axes1[0, 1].imshow(im2_rgb);           axes1[0, 1].set_title("c2 — Target");               axes1[0, 1].axis('off')
axes1[0, 2].imshow(warped_manual_rgb); axes1[0, 2].set_title("(a) c1 Warped → c2");        axes1[0, 2].axis('off')
axes1[1, 0].imshow(im1_pts);           axes1[1, 0].set_title("c1 — 6 Manual Points (red)");axes1[1, 0].axis('off')
axes1[1, 1].imshow(im2_pts);           axes1[1, 1].set_title("c2 — 6 Manual Points (grn)");axes1[1, 1].axis('off')
axes1[1, 2].imshow(diff_manual_amp_rgb);axes1[1, 2].set_title("(b) Diff Image (×3 amp)");  axes1[1, 2].axis('off')

plt.tight_layout()
plt.savefig("q3_ab_manual_homography.png", dpi=130, bbox_inches='tight')
plt.show()

#  Visualization — Figure 2: Part (c) SIFT Matches

fig2, ax2 = plt.subplots(figsize=(16, 6))
fig2.suptitle(f"IT5437 A2 — Q3 Part (c): SIFT Matches (top 40 of {len(good_sift)})",
              fontsize=13, fontweight='bold')
ax2.imshow(match_img_rgb)
ax2.set_title(f"c1: {len(kp1s)} kp  |  c2: {len(kp2s)} kp  |  "
              f"{len(good_sift)} good matches after Lowe's ratio test (0.75)")
ax2.axis('off')
plt.tight_layout()
plt.savefig("q3_c_sift_matches.png", dpi=130, bbox_inches='tight')
plt.show()

#  Visualization — Figure 3: Part (d) SIFT vs Manual Comparison

fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
fig3.suptitle("IT5437 A2 — Q3 Part (d): SIFT Homography vs Manual — Comparison",
              fontsize=13, fontweight='bold')

axes3[0, 0].imshow(warped_manual_rgb);  axes3[0, 0].set_title("(a) Warped — Manual (6 pts)");     axes3[0, 0].axis('off')
axes3[0, 1].imshow(warped_sift_rgb);    axes3[0, 1].set_title(f"(d) Warped — SIFT ({inliers_sift} inliers)"); axes3[0, 1].axis('off')
axes3[0, 2].imshow(im2_rgb);            axes3[0, 2].set_title("c2 — Reference Target");            axes3[0, 2].axis('off')
axes3[1, 0].imshow(diff_manual_amp_rgb);axes3[1, 0].set_title(f"(b) Diff — Manual ×3  (mean={mean_diff_manual:.2f})"); axes3[1, 0].axis('off')
axes3[1, 1].imshow(diff_sift_amp_rgb);  axes3[1, 1].set_title(f"(d) Diff — SIFT   ×3  (mean={mean_diff_sift:.2f})");  axes3[1, 1].axis('off')

# Bar chart: quantitative comparison
ax_bar = axes3[1, 2]
bar_labels  = ['Manual\n(6 pts)', f'SIFT\n({inliers_sift} inliers)']
bar_values  = [mean_diff_manual, mean_diff_sift]
bar_colours = ['#e74c3c', '#3498db']
bars = ax_bar.bar(bar_labels, bar_values, color=bar_colours,
                   width=0.4, edgecolor='black')
ax_bar.set_ylabel("Mean Pixel Difference")
ax_bar.set_title("Alignment Quality\n(lower = better alignment)")
ax_bar.set_ylim(0, max(bar_values) * 1.4)
for bar, val in zip(bars, bar_values):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
ax_bar.grid(True, axis='y', alpha=0.3)
ax_bar.text(0.5, 0.95, f'Winner: {winner}', transform=ax_bar.transAxes,
            ha='center', va='top', fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig("q3_d_sift_comparison.png", dpi=130, bbox_inches='tight')
plt.show()


