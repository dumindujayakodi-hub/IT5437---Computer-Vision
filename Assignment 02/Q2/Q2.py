import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

#  Camera Parameters (given)

f  = 8        # mm  — focal length
do = 720      # mm  — distance from lens to object (earring plane)
p  = 2.2e-3   # mm  — pixel size (2.2 µm converted to mm)


di = (f * do) / (do - f)

print("=" * 55)
print("STEP 1 — Thin Lens Equation")
print("=" * 55)
print(f"  Formula : di = (f * do) / (do - f)")
print(f"  Values  : di = ({f} * {do}) / ({do} - {f})")
print(f"  Result  : di = {di:.5f} mm\n")


m = di / do
print("STEP 2 — Magnification")

print(f"  m = di / do = {di:.5f} / {do} = {m:.6f}")
print(f"  (The image on the sensor is {1/m:.1f}x smaller than the real object)\n")

mm_per_pixel = p / m

print("STEP 3 — Real-world Scale per Pixel")

print(f"  mm/pixel = pixel_size / m")
print(f"           = {p} mm / {m:.6f}")
print(f"           = {mm_per_pixel:.6f} mm/pixel")
print(f"           = {mm_per_pixel:.4f} mm/pixel\n")

#  Step 4: Detect Earrings in Image using HSV Colour Mask
img = cv2.imread("earrings.jpg")
if img is None:
    raise FileNotFoundError("earrings.jpg not found in current directory")

print("STEP 4 — Earring Detection in Image")
print(f"  Image size : {img.shape[1]} x {img.shape[0]} pixels (W x H)")


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_gold = np.array([10, 50, 100])
upper_gold = np.array([40, 255, 255])
mask = cv2.inRange(hsv, lower_gold, upper_gold)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove small noise


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

earrings = sorted([c for c in contours if cv2.contourArea(c) > 3000],
                  key=cv2.contourArea, reverse=True)[:2]

print(f"  Earrings detected : {len(earrings)}\n")



print("STEP 5 — Measurement Results")
results = []
for i, cnt in enumerate(earrings):
    # Bounding rectangle
    x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)

    # Minimum enclosing circle (gives outer diameter)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    outer_d_px = 2 * radius

    # Convert pixel measurements to real-world mm
    real_w = w_b       * mm_per_pixel
    real_h = h_b       * mm_per_pixel
    real_d = outer_d_px * mm_per_pixel

    print(f"  Earring {i+1}:")
    print(f"    Bounding box (pixels)   : {w_b} x {h_b} px")
    print(f"    Enclosing diameter (px) : {outer_d_px:.1f} px")
    print(f"    Real bounding size (mm) : {real_w:.2f} mm x {real_h:.2f} mm")
    print(f"    Real outer diameter(mm) : {real_d:.2f} mm\n")

    results.append({
        'x': x_b, 'y': y_b, 'w': w_b, 'h': h_b,
        'cx': cx, 'cy': cy, 'radius': radius,
        'real_w': real_w, 'real_h': real_h, 'real_d': real_d
    })

#  Visualization

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("IT5437 A2 — Q2: Earring Size Estimation",
             fontsize=14, fontweight='bold')

ax = axes[0]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax.imshow(img_rgb)

ear_colours = ['#e74c3c', '#2ecc71']
for i, r in enumerate(results):
    # Bounding box
    rect = patches.Rectangle((r['x'], r['y']), r['w'], r['h'],
                               linewidth=2, edgecolor=ear_colours[i], facecolor='none')
    # Enclosing circle
    circ = plt.Circle((r['cx'], r['cy']), r['radius'],
                       linewidth=2, edgecolor=ear_colours[i], fill=False, linestyle='--')
    ax.add_patch(rect)
    ax.add_patch(circ)
    ax.text(r['cx'], r['cy'] - r['radius'] - 15,
            f"Dia={r['real_d']:.1f}mm\n{r['real_w']:.1f}x{r['real_h']:.1f}mm",
            ha='center', fontsize=9, color=ear_colours[i],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_title("Detected Earrings with Measurements", fontsize=11)
ax.axis('off')

ax2 = axes[1]
ax2.set_xlim(-50, 200)
ax2.set_ylim(-60, 60)
ax2.set_facecolor('#f8f9fa')

# Optical axis
ax2.axhline(0, color='black', lw=0.8, zorder=1)

# Lens plane at x=0
ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax2.annotate('', xy=(0, -50), xytext=(0, 50),
             arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
ax2.text(3, 45, f'Lens\n(f={f}mm)', fontsize=8, color='gray')

# Object (earring) to the left
ax2.plot([-40, -40], [-25, 25], 'b-', lw=4, label=f'Earring (do={do}mm)', zorder=3)
ax2.text(-40, 30, f'do = {do} mm', ha='center', fontsize=8, color='steelblue')

# Image (sensor) to the right — scaled for diagram
di_scaled = 40  # scaled representation
ax2.plot([di_scaled, di_scaled], [-3, 3], 'r-', lw=4, label=f'Sensor (di={di:.2f}mm)', zorder=3)
ax2.text(di_scaled, 8, f'di = {di:.2f} mm', ha='center', fontsize=8, color='red')

# Ray lines
ax2.plot([-40, di_scaled], [25, -3], 'g--', lw=1.2, alpha=0.6)
ax2.plot([-40, di_scaled], [-25,  3], 'g--', lw=1.2, alpha=0.6)

ax2.set_title("Thin Lens Camera Geometry (not to scale)", fontsize=11)
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.2)
ax2.set_yticks([])
ax2.set_xlabel("Optical Axis")

summary = (
    f"pixel size  = {p*1000:.1f} µm\n"
    f"di          = {di:.3f} mm\n"
    f"magnif. m   = {m:.5f}\n"
    f"scale       = {mm_per_pixel:.4f} mm/px\n"
    f"Earring dia ≈ {results[0]['real_d']:.1f} mm"
)
ax2.text(70, 35, summary, fontsize=8, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig("q2_earring_size.png", dpi=150, bbox_inches='tight')
plt.show()
