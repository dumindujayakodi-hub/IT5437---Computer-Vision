import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#  Load & convert to grayscale 
img_bgr  = cv.imread('Woman.jpg')
assert img_bgr is not None, "Image not found."
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)


# Thresholding

thresh_val, binary = cv.threshold(
    img_gray, 0, 255,
    cv.THRESH_BINARY_INV + cv.THRESH_OTSU   # INV: foreground=dark → white mask
)
print(f"Otsu threshold value: {thresh_val}")


foreground_mask = binary  # uint8, values 0 or 255


# Histogram Equalization ONLY on foreground region

def my_histeq(img: np.ndarray) -> np.ndarray:
   
    hist  = np.zeros(256, dtype=np.int64)
    for val in img.ravel():
        hist[val] += 1
    cdf     = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    N       = img.size
    lut     = np.floor(
        (cdf - cdf_min) / (N - cdf_min) * 255
    ).clip(0, 255).astype(np.uint8)
    lut[cdf == 0] = 0
    return lut[img]



fg_pixels = img_gray[foreground_mask == 255]          # 1-D array of fg pixels


hist  = np.zeros(256, dtype=np.int64)
for v in fg_pixels:
    hist[v] += 1
cdf     = np.cumsum(hist)
cdf_min = cdf[cdf > 0].min()
N       = fg_pixels.size
lut     = np.floor(
    (cdf - cdf_min) / (N - cdf_min) * 255
).clip(0, 255).astype(np.uint8)
lut[cdf == 0] = 0


result = img_gray.copy()
result[foreground_mask == 255] = lut[img_gray[foreground_mask == 255]]


fig, axes = plt.subplots(2, 4, figsize=(18, 9))


imgs_top = [img_gray, foreground_mask, img_gray, result]
titles_top = [
    'Original Grayscale',
    f'Otsu Mask\n(threshold = {int(thresh_val)})',
    'Original (foreground only)',
    'After Selective Equalization'
]
for ax, im, t in zip(axes[0], imgs_top, titles_top):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(t, fontsize=10)
    ax.axis('off')


overlay = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
overlay[foreground_mask == 0] = [0, 0, 180]   # blue = background
axes[0][2].imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
axes[0][2].set_title('Foreground Overlay\n(blue = background)', fontsize=10)
axes[0][2].axis('off')


def plot_hist_region(ax, image, mask_val, mask, title, color):
    px = image[mask == mask_val] if mask is not None else image.ravel()
    ax.hist(px, bins=256, range=(0,255), color=color,
            edgecolor='none', alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 255)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Count')

plot_hist_region(axes[1,0], img_gray, None,  None,              'Hist – Full Original',    'steelblue')
plot_hist_region(axes[1,1], img_gray, 255,   foreground_mask,   'Hist – Foreground Only',  'darkorange')
plot_hist_region(axes[1,2], result,   255,   foreground_mask,   'Hist – After Eq (FG)',    'tomato')
plot_hist_region(axes[1,3], result,   None,  None,              'Hist – Full Result',      'seagreen')


plt.tight_layout()
plt.savefig('q4_results.png', dpi=150)
plt.show()