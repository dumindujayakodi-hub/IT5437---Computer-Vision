import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# Gaussian kernel 

def gaussian_kernel(size, sigma):
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()

# Part (b) – 5×5 Derivative-of-Gaussian kernels for σ = 2

SIGMA = 2
SIZE  = 5
half  = SIZE // 2
ax    = np.arange(-half, half + 1, dtype=np.float64)
xx, yy = np.meshgrid(ax, ax)          # xx[i,j]=x-coord, yy[i,j]=y-coord

G = gaussian_kernel(SIZE, SIGMA)      # base Gaussian

# dG/dx = -(x / σ²) * G(x,y)
dGdx = -(xx / SIGMA**2) * G

# dG/dy = -(y / σ²) * G(x,y)
dGdy = -(yy / SIGMA**2) * G

# Normalize: divide by the sum of absolute values so the filter has unit
# response to a unit-slope ramp
dGdx /= np.abs(dGdx).sum()
dGdy /= np.abs(dGdy).sum()

np.set_printoptions(precision=4, suppress=True)
print("dG/dx kernel:\n", dGdx)
print("\ndG/dy kernel:\n", dGdy)

# Part (c) – Visualize 51×51 dG/dx as 3-D surface

SIZE51 = 51
half51 = SIZE51 // 2
ax51   = np.arange(-half51, half51 + 1, dtype=np.float64)
xx51, yy51 = np.meshgrid(ax51, ax51)

G51    = gaussian_kernel(SIZE51, SIGMA)
dGdx51 = -(xx51 / SIGMA**2) * G51
dGdx51 /= np.abs(dGdx51).sum()

fig3d = plt.figure(figsize=(8, 6))
ax3d  = fig3d.add_subplot(111, projection='3d')
ax3d.plot_surface(xx51, yy51, dGdx51, cmap='RdBu', edgecolor='none', alpha=0.9)
ax3d.set_title(f'51×51 Derivative-of-Gaussian (x-direction, σ={SIGMA})', fontsize=11)
ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('Amplitude')
plt.tight_layout()
plt.savefig('q6.png', dpi=150)
plt.show()

# Part (d) – Apply DoG kernels to get image gradients

img = cv.imread('runway.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found."

# Apply kernels (use float32, ddepth=-1)
k_x = dGdx.astype(np.float32)
k_y = dGdy.astype(np.float32)

grad_x_dog = cv.filter2D(img.astype(np.float32), -1, k_x)
grad_y_dog = cv.filter2D(img.astype(np.float32), -1, k_y)

# Gradient magnitude
mag_dog = np.sqrt(grad_x_dog**2 + grad_y_dog**2)
mag_dog = cv.normalize(mag_dog, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


# Part (e) – OpenCV Sobel for comparison

# Sobel uses a 3×3 kernel by default (ksize=3)
sobel_x = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
sobel_y = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)

mag_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
mag_sobel = cv.normalize(mag_sobel, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Row 0 – DoG
for ax, im, title in zip(axes[0],
    [img,
     np.abs(grad_x_dog).astype(np.uint8),
     np.abs(grad_y_dog).astype(np.uint8),
     mag_dog],
    ['Original',
     'DoG – Grad X',
     'DoG – Grad Y',
     'DoG – Gradient Magnitude']):
    ax.imshow(im, cmap='gray'); ax.set_title(title, fontsize=10); ax.axis('off')

# Row 1 – Sobel
for ax, im, title in zip(axes[1],
    [img,
     cv.convertScaleAbs(sobel_x),
     cv.convertScaleAbs(sobel_y),
     mag_sobel],
    ['Original',
     'Sobel – Grad X',
     'Sobel – Grad Y',
     'Sobel – Gradient Magnitude']):
    ax.imshow(im, cmap='gray'); ax.set_title(title, fontsize=10); ax.axis('off')

plt.suptitle("Q6: Derivative of Gaussian vs Sobel", fontsize=13)
plt.tight_layout()
plt.savefig('q6_results.png', dpi=150)
plt.show()