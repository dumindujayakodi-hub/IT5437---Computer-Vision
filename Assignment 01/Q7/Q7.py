import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Core zoom function

def zoom_image(img: np.ndarray, s: float, method: str = 'bilinear') -> np.ndarray:
    
    assert 0 < s <= 10, "s must be in (0, 10]"
    assert method in ('nearest', 'bilinear'), "method must be 'nearest' or 'bilinear'"

    H, W = img.shape[:2]
    out_H = int(round(H * s))
    out_W = int(round(W * s))

    # Handle colour (3-ch) by processing each channel
    is_color = img.ndim == 3
    if is_color:
        channels = [zoom_image(img[:, :, c], s, method) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)

    # For each output pixel (r_out, c_out) find the source coordinate
    # using the inverse mapping:  src = out / s  (centre-aligned)
    r_out, c_out = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')

    # Map output coords → source coords (float)
    r_src = (r_out + 0.5) / s - 0.5
    c_src = (c_out + 0.5) / s - 0.5

    if method == 'nearest':
        # Round to nearest integer, clip to valid range
        r_nn = np.clip(np.round(r_src).astype(int), 0, H - 1)
        c_nn = np.clip(np.round(c_src).astype(int), 0, W - 1)
        return img[r_nn, c_nn].astype(np.uint8)

    else:  # bilinear
        # Floor coordinates → top-left neighbour
        r0 = np.clip(np.floor(r_src).astype(int), 0, H - 2)
        c0 = np.clip(np.floor(c_src).astype(int), 0, W - 2)
        r1 = r0 + 1
        c1 = c0 + 1

        # Fractional offsets
        dr = (r_src - r0).clip(0, 1)
        dc = (c_src - c0).clip(0, 1)

        # Bilinear interpolation
        img_f = img.astype(np.float32)
        p00 = img_f[r0, c0]
        p10 = img_f[r1, c0]
        p01 = img_f[r0, c1]
        p11 = img_f[r1, c1]

        out = ((1 - dr) * (1 - dc) * p00 +
                    dr  * (1 - dc) * p10 +
               (1 - dr) *      dc  * p01 +
                    dr  *      dc  * p11)

        return np.clip(out, 0, 255).astype(np.uint8)


# Normalized SSD

def normalized_ssd(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Normalized Sum of Squared Differences.
    Both images must have the same shape.
    Normalized by the number of pixels so it is scale-independent.
    """
    assert img_a.shape == img_b.shape, "Images must be the same size."
    diff = img_a.astype(np.float64) - img_b.astype(np.float64)
    return np.sum(diff ** 2) / diff.size


# Demo: zoom a small image up, compare to the large original

# Load large original and small (zoomed-out) version
large_orig = cv.imread('original_large.jpg',  cv.IMREAD_GRAYSCALE)
small_img  = cv.imread('small_version.jpg',   cv.IMREAD_GRAYSCALE)
assert large_orig is not None and small_img is not None, "Check file names."

# Compute zoom factor from image sizes
H_large, W_large = large_orig.shape
H_small, W_small = small_img.shape
s = H_large / H_small           # assume aspect ratio is preserved
print(f"Zoom factor s = {s:.3f}")

# Scale up small image using both methods
zoomed_nn  = zoom_image(small_img, s, method='nearest')
zoomed_bil = zoom_image(small_img, s, method='bilinear')

# Resize to exactly match large original dimensions (handles rounding)
zoomed_nn  = cv.resize(zoomed_nn,  (W_large, H_large), interpolation=cv.INTER_NEAREST)
zoomed_bil = cv.resize(zoomed_bil, (W_large, H_large), interpolation=cv.INTER_NEAREST)

# SSD evaluation
ssd_nn  = normalized_ssd(large_orig, zoomed_nn)
ssd_bil = normalized_ssd(large_orig, zoomed_bil)
print(f"Normalized SSD  – Nearest-Neighbour : {ssd_nn:.2f}")
print(f"Normalized SSD  – Bilinear          : {ssd_bil:.2f}")


fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, im, title in zip(axes,
    [large_orig, zoomed_nn, zoomed_bil,
     np.abs(zoomed_bil.astype(int) - large_orig.astype(int)).astype(np.uint8)],
    ['Large Original',
     f'Nearest-Neighbour\nSSD={ssd_nn:.1f}',
     f'Bilinear\nSSD={ssd_bil:.1f}',
     'Diff |Bilinear − Original|']):
    ax.imshow(im, cmap='gray'); ax.set_title(title, fontsize=10); ax.axis('off')


plt.tight_layout()
plt.savefig('q7_results.png', dpi=150)
plt.show()


test_img = cv.imread('runway.jpg', cv.IMREAD_GRAYSCALE)
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
for i, s_test in enumerate([0.5, 1.5, 2.0, 3.0]):
    axes2[0, i].imshow(zoom_image(test_img, s_test, 'nearest'),
                       cmap='gray'); axes2[0, i].axis('off')
    axes2[0, i].set_title(f'NN  s={s_test}')
    axes2[1, i].imshow(zoom_image(test_img, s_test, 'bilinear'),
                       cmap='gray'); axes2[1, i].axis('off')
    axes2[1, i].set_title(f'Bilinear  s={s_test}')


plt.tight_layout()
plt.savefig('q7_zoom_grid.png', dpi=150)
plt.show()