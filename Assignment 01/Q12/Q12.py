import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Part (c) – Homomorphic Filtering Algorithm

def homomorphic_filter(img_gray: np.ndarray,
                       sigma: float  = 30.0,
                       gamma_l: float = 0.3,
                       gamma_h: float = 1.8,
                       c: float       = 1.0) -> np.ndarray:
    
    img_f = img_gray.astype(np.float64)

    # Log transform (add 1 to avoid log(0)) 
    z = np.log1p(img_f)          # log1p(x) = ln(x+1), numerically stable

    # 2D FFT (shift DC to centre)
    Z = np.fft.fftshift(np.fft.fft2(z))

    # High-Emphasis Gaussian Filter H(u,v) 
    # H = (gamma_h - gamma_l) * [1 - exp(-c * D²/sigma²)] + gamma_l
    # At low freq (D≈0): H ≈ gamma_l  (suppress illumination)
    # At high freq (D large): H ≈ gamma_h  (boost reflectance)
    H_img, W_img = img_gray.shape
    cy, cx = H_img // 2, W_img // 2
    Y, X   = np.ogrid[:H_img, :W_img]
    D2     = (X - cx)**2 + (Y - cy)**2         # squared distance from DC

    H = (gamma_h - gamma_l) * \
        (1 - np.exp(-c * D2 / (sigma**2))) + gamma_l

    # Apply filter in frequency domain 
    S = H * Z

    # Inverse FFT
    s = np.real(np.fft.ifft2(np.fft.ifftshift(S)))

    # Exponential (inverse of log) 
    g = np.expm1(s)              # expm1(x) = exp(x)-1, inverse of log1p

    # Normalize to [0, 255]
    g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)
    return g.astype(np.uint8)



# Apply to an image of choice

img_bgr  = cv.imread('runway.png')
assert img_bgr is not None, "Image not found."
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# Apply with different parameter sets
result_mild   = homomorphic_filter(img_gray, sigma=30, gamma_l=0.5,
                                   gamma_h=1.5, c=1.0)
result_strong = homomorphic_filter(img_gray, sigma=20, gamma_l=0.3,
                                   gamma_h=2.0, c=1.0)

# Histogram equalization for comparison 
histeq = cv.equalizeHist(img_gray)

H_img, W_img = img_gray.shape
cy, cx = H_img // 2, W_img // 2
Y, X   = np.ogrid[:H_img, :W_img]
D2     = (X - cx)**2 + (Y - cy)**2
H_vis  = (2.0 - 0.3) * (1 - np.exp(-1.0 * D2 / (30.0**2))) + 0.3

# Cross-section of the filter
mid       = H_img // 2
freq_axis = np.linspace(-0.5, 0.5, W_img)



# Figure 1 – Filter shape
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
axes1[0].imshow(H_vis, cmap='hot')
axes1[0].set_title('H(u,v) – High-Emphasis Filter\n(dark=low gain, bright=high gain)',
                   fontsize=10)
axes1[0].axis('off')
axes1[1].plot(freq_axis, H_vis[mid, :], color='tomato', linewidth=2)
axes1[1].axhline(1.0, color='gray', linestyle='--', linewidth=1)
axes1[1].set_title('H(u,v) Cross-Section\n(γL=0.3 at DC, γH=2.0 at high freq)',
                   fontsize=10)
axes1[1].set_xlabel('Normalized Frequency')
axes1[1].set_ylabel('Gain')
axes1[1].set_xlim(-0.5, 0.5)
axes1[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('q12_filter.png', dpi=150)
plt.show()

# Figure 2 – Image results
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
results = [
    (img_gray,      'Original'),
    (result_mild,   'Homomorphic\n(mild: γL=0.5, γH=1.5)'),
    (result_strong, 'Homomorphic\n(strong: γL=0.3, γH=2.0)'),
    (histeq,        'Hist. Equalization\n(for comparison)'),
]
for i, (im, title) in enumerate(results):
    axes2[0, i].imshow(im, cmap='gray', vmin=0, vmax=255)
    axes2[0, i].set_title(title, fontsize=10)
    axes2[0, i].axis('off')
    axes2[1, i].hist(im.ravel(), bins=256, range=(0,255),
                     color='steelblue', edgecolor='none')
    axes2[1, i].set_xlim(0, 255)
    axes2[1, i].set_xlabel('Intensity')
    axes2[1, i].set_ylabel('Count')
    axes2[1, i].set_title(f'Histogram – {title.splitlines()[0]}', fontsize=9)


plt.tight_layout()
plt.savefig('q12_results.png', dpi=150)
plt.show()