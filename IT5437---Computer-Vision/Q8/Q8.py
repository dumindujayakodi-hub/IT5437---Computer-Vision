import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load noisy image 
img = cv.imread('emma1.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found."


def add_salt_pepper(img, prob=0.05):
    """
    Add salt & pepper noise to a grayscale image.
    prob : fraction of pixels to corrupt (e.g. 0.05 = 5%)
    """
    noisy = img.copy()
    rng   = np.random.default_rng(42)
    mask  = rng.random(img.shape)
    noisy[mask < prob / 2]         = 0    # pepper
    noisy[mask > 1 - prob / 2]     = 255  # salt
    return noisy

# Part (a) 


gauss_blur = cv.GaussianBlur(img, (5, 5), sigmaX=1.5)

# Also try a larger kernel to show trade-off
gauss_blur_large = cv.GaussianBlur(img, (9, 9), sigmaX=2.5)


# Part (b) – Median Filtering
median_3 = cv.medianBlur(img, 3)
median_5 = cv.medianBlur(img, 5)   # stronger, for heavier noise

def psnr(ref, img):
    mse = np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


# Plotting

results = [
    (img,              'Noisy Input'),
    (gauss_blur,       'Gaussian 5×5\n(σ=1.5)'),
    (gauss_blur_large, 'Gaussian 9×9\n(σ=2.5)'),
    (median_3,         'Median 3×3'),
    (median_5,         'Median 5×5'),
]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, (im, title) in enumerate(results):
    # Image row
    axes[0, i].imshow(im, cmap='gray', vmin=0, vmax=255)
    axes[0, i].set_title(title, fontsize=10)
    axes[0, i].axis('off')
    # Histogram row
    axes[1, i].hist(im.ravel(), bins=256, range=(0,255),
                    color='steelblue', edgecolor='none')
    axes[1, i].set_xlim(0, 255)
    axes[1, i].set_xlabel('Intensity')
    axes[1, i].set_ylabel('Count')
    axes[1, i].set_title(f'Histogram – {title.splitlines()[0]}', fontsize=9)


plt.tight_layout()
plt.savefig('q8_results.png', dpi=150)
plt.show()

# Zoomed crop for close-up comparison
H, W  = img.shape
r1, r2, c1, c2 = H//4, 3*H//4, W//4, 3*W//4   # centre crop

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 5))
crops = [(img, 'Noisy'), (gauss_blur, 'Gaussian 5×5'),
         (median_3, 'Median 3×3'), (median_5, 'Median 5×5')]
for ax, (im, title) in zip(axes2, crops):
    ax.imshow(im[r1:r2, c1:c2], cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=11); ax.axis('off')


plt.tight_layout()
plt.savefig('q8_crop.png', dpi=150)
plt.show()