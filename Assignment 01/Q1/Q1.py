import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image 
img = cv.imread("runway.png", cv.IMREAD_GRAYSCALE)
assert img is not None, 'image not found. Check the file path.'

# Normalize to [0, 1] float for processing
img_norm = img.astype(np.float32) / 255.0

#(a) Gamma correction γ = 0.5  (brightens the image)
gamma_05 = np.power(img_norm, 0.5)

#(b) Gamma correction γ = 2  (darkens the image) 
gamma_2 = np.power(img_norm, 2.0)

#(c) Contrast Stretching 
r1, r2 = 0.2, 0.8

def contrast_stretch(r_img, r1, r2):
  
    s = np.zeros_like(r_img)                          # default: 0
    mid = (r_img >= r1) & (r_img <= r2)               # middle band mask
    s[mid] = (r_img[mid] - r1) / (r2 - r1)            # linear remap
    s[r_img > r2] = 1.0                                # clip high
    return s

contrast = contrast_stretch(img_norm, r1, r2)

# Display results 
titles = [
    'Original',
    'Gamma γ=0.5 (brighter)',
    'Gamma γ=2.0 (darker)',
    f'Contrast Stretch\nr1={r1}, r2={r2}'
]
images = [img_norm, gamma_05, gamma_2, contrast]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, (ax_img, ax_hist, title, im) in enumerate(
        zip(axes[0], axes[1], titles, images)):
    
    # Image row
    ax_img.imshow(im, cmap='gray', vmin=0, vmax=1)
    ax_img.set_title(title, fontsize=11)
    ax_img.axis('off')

    # Histogram row
    ax_hist.hist(im.ravel(), bins=256, range=(0, 1), color='steelblue',
                 edgecolor='none')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel('Intensity')
    ax_hist.set_ylabel('Pixel Count')
    ax_hist.set_title(f'Histogram – {title.splitlines()[0]}', fontsize=9)

plt.tight_layout()
plt.savefig('q1_results.png', dpi=150)
plt.show()
