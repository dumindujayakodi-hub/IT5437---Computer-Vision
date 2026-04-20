import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# Helper: compute & visualise frequency response of a kernel

def freq_response(kernel: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Zero-pad kernel to (size x size) and compute magnitude of its 2D DFT.
    Returns log-scaled magnitude spectrum (shifted so DC is at centre).
    """
    k_pad = np.zeros((size, size), dtype=np.float64)
    kh, kw = kernel.shape
    # Place kernel at top-left corner before FFT
    k_pad[:kh, :kw] = kernel
    F     = np.fft.fftshift(np.fft.fft2(k_pad))
    mag   = np.abs(F)
    return mag / mag.max()          # normalize to [0,1]



# Define the three filters

SIZE = 256

# 1. Box (averaging) filter – 5×5
box = np.ones((5, 5), dtype=np.float64) / 25.0

# 2. Gaussian filter – 5×5, σ=2
def make_gaussian(size, sigma):
    half = size // 2
    ax   = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()

gauss = make_gaussian(5, 2)

# 3. Laplacian filter – 3×3 (8-connected)
laplacian = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=np.float64)

# Frequency responses
H_box  = freq_response(box,      SIZE)
H_gaus = freq_response(gauss,    SIZE)
H_lap  = freq_response(laplacian, SIZE)

# Part (b) – Visualise frequency responses (2D + 1D cross-section)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

labels   = ['Box Filter (5×5)', 'Gaussian (5×5, σ=2)', 'Laplacian (3×3)']
spectra  = [H_box, H_gaus, H_lap]
cmaps    = ['hot', 'hot', 'RdBu']

for i, (H, label, cmap) in enumerate(zip(spectra, labels, cmaps)):
    # 2D frequency map
    axes[0, i].imshow(H, cmap=cmap, vmin=0, vmax=1)
    axes[0, i].set_title(f'Freq. Response\n{label}', fontsize=10)
    axes[0, i].axis('off')

    # 1D horizontal cross-section through centre
    mid  = SIZE // 2
    freq = np.linspace(-0.5, 0.5, SIZE)   # normalized frequency axis
    axes[1, i].plot(freq, H[mid, :], color='steelblue', linewidth=1.8)
    axes[1, i].set_title(f'Cross-section\n{label}', fontsize=10)
    axes[1, i].set_xlabel('Normalized Frequency')
    axes[1, i].set_ylabel('|H(u)|')
    axes[1, i].set_xlim(-0.5, 0.5)
    axes[1, i].set_ylim(0, 1.05)
    axes[1, i].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1, i].grid(True, alpha=0.3)

plt.suptitle("Q11: Frequency Responses of Spatial Filters", fontsize=13)
plt.tight_layout()
plt.savefig('q11_freq_response.png', dpi=150)
plt.show()


# Part (c) – Ideal low-pass vs Gaussian: ringing comparison

img = cv.imread('runway.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found."

H_img, W_img = img.shape
img_f = img.astype(np.float64)

# Compute image FFT
F_img   = np.fft.fftshift(np.fft.fft2(img_f))

# Ideal Low-Pass Filter 
cutoff  = 30    # pixels from DC centre
cy, cx  = H_img // 2, W_img // 2
Y, X    = np.ogrid[:H_img, :W_img]
dist    = np.sqrt((X - cx)**2 + (Y - cy)**2)
ideal_lpf = (dist <= cutoff).astype(np.float64)   # 1 inside, 0 outside

# Apply ideal LPF
F_ideal   = F_img * ideal_lpf
img_ideal = np.real(np.fft.ifft2(np.fft.ifftshift(F_ideal)))
img_ideal = np.clip(img_ideal, 0, 255).astype(np.uint8)

# Gaussian Low-Pass Filter (smooth falloff in frequency domain)
sigma_freq = 30   # controls bandwidth
gauss_lpf  = np.exp(-dist**2 / (2 * sigma_freq**2))

F_gauss    = F_img * gauss_lpf
img_gauss  = np.real(np.fft.ifft2(np.fft.ifftshift(F_gauss)))
img_gauss  = np.clip(img_gauss, 0, 255).astype(np.uint8)

# Plot ringing comparison 
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
for ax, im, title in zip(axes2,
    [img, img_ideal, img_gauss],
    ['Original',
     f'Ideal LPF\n(cutoff={cutoff}px) — ringing!',
     f'Gaussian LPF\n(σ={sigma_freq}px) — smooth']):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

plt.suptitle("Q11(c): Ringing — Ideal LPF vs Gaussian LPF", fontsize=13)
plt.tight_layout()
plt.savefig('q11_ringing.png', dpi=150)
plt.show()


# Part (d) – Apply all three filters and compare on noisy image

# Add Gaussian noise for this test
rng       = np.random.default_rng(0)
noisy     = np.clip(img_f + rng.normal(0, 25, img_f.shape), 0, 255).astype(np.uint8)

box_filt  = cv.filter2D(noisy, -1, box.astype(np.float32))
gaus_filt = cv.GaussianBlur(noisy, (5, 5), sigmaX=2)
lap_filt  = cv.filter2D(noisy.astype(np.float32), -1,
                         laplacian.astype(np.float32))

# Laplacian sharpens 
lap_disp  = cv.convertScaleAbs(lap_filt)

fig3, axes3 = plt.subplots(1, 4, figsize=(18, 5))
for ax, im, title in zip(axes3,
    [noisy, box_filt, gaus_filt, lap_disp],
    ['Noisy Input',
     'Box Filter\n(low-pass, blocky)',
     'Gaussian Filter\n(low-pass, smooth)',
     'Laplacian\n(high-pass, edges only)']):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


plt.tight_layout()
plt.savefig('q11_noise.png', dpi=150)
plt.show()