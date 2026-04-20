import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image (colour) 
img_bgr = cv.imread('image2.jpg')
assert img_bgr is not None, "Image not found."
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# Convert to L*a*b* 
# OpenCV L*a*b*: L in [0,255], a/b in [0,255] (shifted from [-128,127])
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab).astype(np.float32)

L, a, b = cv.split(img_lab)

# Gamma correction on L only
GAMMA = 0.6   # <-- chosen because the image is underexposed / dark

# L is in [0, 255]; normalize → apply gamma → scale back
L_norm      = L / 255.0
L_corrected = np.power(L_norm, GAMMA) * 255.0
L_corrected = np.clip(L_corrected, 0, 255).astype(np.float32)

# Merge and convert back to RGB 
lab_corrected = cv.merge([L_corrected, a, b]).astype(np.uint8)
img_corrected_bgr = cv.cvtColor(lab_corrected, cv.COLOR_Lab2BGR)
img_corrected_rgb = cv.cvtColor(img_corrected_bgr, cv.COLOR_BGR2RGB)

#  histogram of an RGB image (combined luminance proxy) 
def plot_hist(ax, img_rgb, title, color='steelblue'):
    gray = cv.cvtColor(cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR),
                       cv.COLOR_BGR2GRAY)
    ax.hist(gray.ravel(), bins=256, range=(0, 255),
            color=color, edgecolor='none', alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 255)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Pixel Count')

# Plot: images + histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Images (top row)
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original Image', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(img_corrected_rgb)
axes[0, 1].set_title(f'Gamma Corrected (γ = {GAMMA})\non L* channel only', fontsize=11)
axes[0, 1].axis('off')

# Histograms (bottom row)
plot_hist(axes[1, 0], img_rgb,           'Histogram – Original',     color='steelblue')
plot_hist(axes[1, 1], img_corrected_rgb, f'Histogram – γ={GAMMA}',   color='tomato')

plt.tight_layout()
plt.savefig('q2_results.png', dpi=150)
plt.show()
print(f"Gamma applied: γ = {GAMMA}")
print("Results saved to q2_results.png")