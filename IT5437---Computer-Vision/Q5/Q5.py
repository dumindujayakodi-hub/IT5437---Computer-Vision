import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (needed for 3D projection)


# Helper: build a normalized 2-D Gaussian kernel of any size

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create a normalized (size x size) Gaussian kernel.
    'size' should be odd so there is a clear centre pixel.
    """
    half = size // 2
    # coordinate grid  e.g. for size=5: [-2,-1,0,1,2]
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)          # 2-D coordinate grids

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()                # normalize so values sum to 1
    return kernel


# Part (a) – 5×5 kernel for σ = 2

SIGMA = 2
k5 = gaussian_kernel(5, SIGMA)
print("5×5 Gaussian kernel (σ=2):")
np.set_printoptions(precision=4, suppress=True)
print(k5)
print(f"Sum of kernel = {k5.sum():.6f}  (should be 1.0)")


# Part (b) – Visualize 51×51 kernel as a 3-D surface

k51 = gaussian_kernel(51, SIGMA)

half51 = 25
ax_vals = np.arange(-half51, half51 + 1)
X, Y   = np.meshgrid(ax_vals, ax_vals)

fig3d = plt.figure(figsize=(8, 6))
ax3d  = fig3d.add_subplot(111, projection='3d')
ax3d.plot_surface(X, Y, k51, cmap='viridis', edgecolor='none', alpha=0.9)
ax3d.set_title(f'51×51 Gaussian Kernel  (σ={SIGMA})', fontsize=12)
ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('Amplitude')
plt.tight_layout()
plt.savefig('q5_kernel_3d.png', dpi=150)
plt.show()


# Part (c) & (d) – Apply smoothing: manual kernel vs cv.GaussianBlur

img = cv.imread('runway.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found."

# (c) Manual convolution using the 5×5 kernel we built
#     cv.filter2D applies a custom kernel — same as manual convolution
k5_f32 = k5.astype(np.float32)
manual_blur = cv.filter2D(img, -1, k5_f32)   # -1 = same depth as source

# (d) OpenCV built-in GaussianBlur
#     ksize must be odd; sigmaX=sigmaY=SIGMA
cv_blur = cv.GaussianBlur(img, (5, 5), sigmaX=SIGMA, sigmaY=SIGMA)

# Difference between the two
diff = cv.absdiff(manual_blur, cv_blur)

# Plot results 
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for ax, im, title in zip(axes,
    [img, manual_blur, cv_blur, diff],
    ['Original',
     'Manual Gaussian (5×5, σ=2)',
     'cv.GaussianBlur (5×5, σ=2)',
     f'|Manual − OpenCV|\nmax diff = {diff.max()}']):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.suptitle("Q5: Gaussian Smoothing", fontsize=13)
plt.tight_layout()
plt.savefig('q5_results.png', dpi=150)
plt.show()

# Print kernel 
print("\nKernel values for report (multiply by 1 to see weights):")
print(k5)