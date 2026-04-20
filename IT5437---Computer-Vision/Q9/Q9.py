import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('runway.png', cv.IMREAD_GRAYSCALE)   # use any image of choice
assert img is not None, "Image not found."
img_f = img.astype(np.float32)


# Method 1 – Unsharp Masking

def unsharp_mask(img_f, sigma=1.5, k=1.5):
    """
    img_f : float32 grayscale image
    sigma : Gaussian blur strength (controls which frequencies are boosted)
    k     : sharpening strength  (higher = more aggressive)
    """
    blurred = cv.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
    # High-frequency detail (edge map)
    detail  = img_f - blurred
    # Add scaled detail back
    sharp   = img_f + k * detail
    return np.clip(sharp, 0, 255).astype(np.uint8), \
           np.clip(detail + 128, 0, 255).astype(np.uint8)  # for visualisation

sharp_usm_mild,   detail_mild   = unsharp_mask(img_f, sigma=1.5, k=1.0)
sharp_usm_strong, detail_strong = unsharp_mask(img_f, sigma=1.5, k=2.5)


# Method 2 – Laplacian Sharpening


# 3×3 Laplacian kernel (4-connected)
laplacian_kernel = np.array([[0, -1,  0],
                              [-1,  4, -1],
                              [0, -1,  0]], dtype=np.float32)

# 3×3 Laplacian kernel (8-connected, more sensitive)
laplacian_kernel_8 = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=np.float32)

lap      = cv.filter2D(img_f, -1, laplacian_kernel)
lap_sharp = np.clip(img_f + lap, 0, 255).astype(np.uint8)

lap8      = cv.filter2D(img_f, -1, laplacian_kernel_8)
lap8_sharp = np.clip(img_f + lap8, 0, 255).astype(np.uint8)


# Method 3 – OpenCV built-in sharpening kernel (for comparison)

sharpen_kernel = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]], dtype=np.float32)
cv_sharp = cv.filter2D(img, -1, sharpen_kernel)


# Plotting
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

imgs = [
    (img,            'Original'),
    (sharp_usm_mild, 'Unsharp Mask\n(σ=1.5, k=1.0)'),
    (sharp_usm_strong,'Unsharp Mask\n(σ=1.5, k=2.5)'),
    (detail_strong,  'Edge Detail\n(boosted for display)'),
    (lap_sharp,      'Laplacian Sharp\n(4-connected)'),
    (lap8_sharp,     'Laplacian Sharp\n(8-connected)'),
    (cv_sharp,       'CV Sharpen Kernel'),
    (np.abs(sharp_usm_strong.astype(int) -
            img.astype(int)).astype(np.uint8),
                     'Difference\n|USM − Original|'),
]

for ax, (im, title) in zip(axes.ravel(), imgs):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


plt.tight_layout()
plt.savefig('q9_results.png', dpi=150)
plt.show()


H, W = img.shape
r1, r2 = H//3,   2*H//3
c1, c2 = W//3,   2*W//3

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 5))
crops = [
    (img,             'Original (crop)'),
    (sharp_usm_mild,  'USM k=1.0 (crop)'),
    (sharp_usm_strong,'USM k=2.5 (crop)'),
    (lap8_sharp,      'Laplacian 8 (crop)'),
]
for ax, (im, title) in zip(axes2, crops):
    ax.imshow(im[r1:r2, c1:c2], cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


plt.tight_layout()
plt.savefig('q9_crop.png', dpi=150)
plt.show()