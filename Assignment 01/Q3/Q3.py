import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load runway image
img = cv.imread('runway.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found."

# Custom Histogram Equalization 

def my_histeq(img: np.ndarray) -> np.ndarray:
    """
    Manually equalize the histogram of a uint8 grayscale image.

    Steps:
      1. Compute histogram (256 bins).
      2. Compute CDF from the histogram.
      3. Normalize CDF to create a lookup table [0, 255].
      4. Remap every pixel through the lookup table.
    """
    assert img.dtype == np.uint8, "Input must be uint8 grayscale."

   
    hist = np.zeros(256, dtype=np.int64)
    for val in img.ravel():         
        hist[val] += 1

    cdf = np.cumsum(hist)            # shape (256,)

  
    cdf_min = cdf[cdf > 0].min()     
    N       = img.size               

    lut = np.floor(
        (cdf - cdf_min) / (N - cdf_min) * 255
    ).clip(0, 255).astype(np.uint8)

    lut[cdf == 0] = 0


    eq_img = lut[img]               
    return eq_img, hist, cdf, lut


eq_img, hist, cdf, lut = my_histeq(img)


eq_ref = cv.equalizeHist(img)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Row 0 – images
for ax, im, title in zip(axes[0],
                          [img, eq_img, eq_ref],
                          ['Original', 'My equalizeHist', 'cv.equalizeHist (ref)']):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

# Row 1 – histograms
bins = np.arange(256)
axes[1, 0].bar(bins, hist, color='steelblue', width=1)
axes[1, 0].set_title('Histogram – Original')
axes[1, 0].set_xlim(0, 255)

hist_eq, _ = np.histogram(eq_img, 256, [0, 256])
axes[1, 1].bar(bins, hist_eq, color='tomato', width=1)
axes[1, 1].set_title('Histogram – My equalizeHist')
axes[1, 1].set_xlim(0, 255)

hist_ref, _ = np.histogram(eq_ref, 256, [0, 256])
axes[1, 2].bar(bins, hist_ref, color='seagreen', width=1)
axes[1, 2].set_title('Histogram – cv.equalizeHist (ref)')
axes[1, 2].set_xlim(0, 255)

# Row 2 – CDF and LUT
axes[2, 0].plot(bins, cdf / cdf.max(), color='steelblue')
axes[2, 0].set_title('Normalized CDF')
axes[2, 0].set_xlim(0, 255)
axes[2, 0].set_xlabel('Intensity')

axes[2, 1].plot(bins, lut, color='tomato')
axes[2, 1].set_title('Lookup Table (LUT)')
axes[2, 1].set_xlim(0, 255)
axes[2, 1].set_xlabel('Input Intensity')
axes[2, 1].set_ylabel('Output Intensity')


diff = cv.absdiff(eq_img, eq_ref)
axes[2, 2].imshow(diff, cmap='hot')
axes[2, 2].set_title(f'Diff |mine − cv| (max={diff.max()})')
axes[2, 2].axis('off')


plt.tight_layout()
plt.savefig('q3_results.png', dpi=150)
plt.show()
