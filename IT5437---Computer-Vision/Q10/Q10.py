import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Part (a) – Manual Bilateral Filter

def bilateral_filter(img: np.ndarray,
                     diameter: int,
                     sigma_s: float,
                     sigma_r: float) -> np.ndarray:
    
    assert img.ndim == 2,        "Input must be grayscale."
    assert diameter % 2 == 1,   "Diameter must be odd."

    img_f  = img.astype(np.float64)
    radius = diameter // 2
    H, W   = img.shape
    out    = np.zeros_like(img_f)

    # Pre-compute spatial Gaussian weights for the kernel window
    ax  = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    spatial_w = np.exp(-(xx**2 + yy**2) / (2 * sigma_s**2))  # shape (d, d)

    # Pad image to handle borders
    img_pad = np.pad(img_f, radius, mode='reflect')

    for r in range(H):
        for c in range(W):
            # Extract local patch
            patch = img_pad[r: r + diameter,
                            c: c + diameter]          # shape (d, d)

            # Range (intensity) Gaussian weight
            diff    = patch - img_f[r, c]
            range_w = np.exp(-(diff**2) / (2 * sigma_r**2))

            # Combined bilateral weight
            w = spatial_w * range_w                   # element-wise

            # Normalized weighted sum
            out[r, c] = np.sum(w * patch) / np.sum(w)

    return np.clip(out, 0, 255).astype(np.uint8)


# Load image 
img = cv.imread('runway.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found."

# Use a smaller crop for the manual filter 
H, W = img.shape
crop = img[H//4: 3*H//4, W//4: 3*W//4]   # centre crop to save time

print("Running manual bilateral filter (this may take ~30s on a large crop)…")
D, SS, SR = 9, 15, 40           # diameter, sigma_s, sigma_r
manual_bil = bilateral_filter(crop, diameter=D, sigma_s=SS, sigma_r=SR)
print("Done.")


# Part (b) – Gaussian Blur (OpenCV reference)

gauss_blur = cv.GaussianBlur(img, (D, D), sigmaX=SS)


# Part (c) – OpenCV bilateral filter (on full image + crop for comparison)

# cv.bilateralFilter(src, d, sigmaColor, sigmaSpace)
# sigmaColor = sigma_r,  sigmaSpace = sigma_s
cv_bil_full = cv.bilateralFilter(img,  d=D, sigmaColor=SR, sigmaSpace=SS)
cv_bil_crop = cv.bilateralFilter(crop, d=D, sigmaColor=SR, sigmaSpace=SS)


# Part (d) – Compare manual vs OpenCV bilateral (on the crop)

diff_crop = cv.absdiff(manual_bil, cv_bil_crop)
print(f"Max pixel diff  manual vs cv.bilateralFilter: {diff_crop.max()}")
print(f"Mean pixel diff manual vs cv.bilateralFilter: {diff_crop.mean():.3f}")


# Plotting – full image comparison

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, im, title in zip(axes,
    [img, gauss_blur, cv_bil_full],
    ['Original',
     f'Gaussian Blur\n(ksize={D}, σ={SS})',
     f'cv.bilateralFilter\n(d={D}, σs={SS}, σr={SR})']):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


plt.tight_layout()
plt.savefig('q10_full.png', dpi=150)
plt.show()

# Crop comparison: manual vs OpenCV bilateral 
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
for ax, im, title in zip(axes2,
    [crop, manual_bil, cv_bil_crop, diff_crop],
    ['Original Crop',
     f'Manual Bilateral\n(d={D}, σs={SS}, σr={SR})',
     f'cv.bilateralFilter\n(d={D}, σs={SS}, σr={SR})',
     f'|Manual − CV|\nmax={diff_crop.max()}']):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


plt.tight_layout()
plt.savefig('q10_crop.png', dpi=150)
plt.show()

# Edge preservation demo 

def edge_strength(im):
    sx = cv.Sobel(im, cv.CV_32F, 1, 0, ksize=3)
    sy = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    return cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
for ax, im, title in zip(axes3,
    [edge_strength(img),
     edge_strength(gauss_blur),
     edge_strength(cv_bil_full)],
    ['Edge Map – Original',
     'Edge Map – Gaussian',
     'Edge Map – Bilateral']):
    ax.imshow(im, cmap='hot')
    ax.set_title(title, fontsize=11)
    ax.axis('off')


plt.tight_layout()
plt.savefig('q10_edges.png', dpi=150)
plt.show()