import sys
import matplotlib.pyplot as plt
from matplotlib.style import use
from skimage import io, color
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from Tp1_Math_Functions import create_output_directory, energy_N_L, read_image, to_grayscale, energy 
from Tp1_Math_Functions import to_float, get_output_filename, plot_image, resize_image
from Errors import mse, mae, emax, psnr, laplacian

SORTIES = create_output_directory()
u = read_image(sys.argv[1])
u = to_grayscale(u)
u = to_float(u)

# Resized image 512x512
u_r = resize_image(u, 512, 512)
n = u_r.size

# Generate Gaussian noise
rng = np.random.default_rng(0)
# sigs = [20, 30, 60, 190]

# plot the noisy images for different sigma values
# plt.figure(figsize=(15, 4))
# for sig in sigs:
#     b = sig * rng.standard_normal(u_r.shape)
#     f = u_r + b                                             # add the Gaussian noise
#     plt.subplot(1, len(sigs), sigs.index(sig)+1)
#     plt.imshow(f, cmap="gray", vmin=0, vmax=255)
#     plt.title(f"Sigma = {sig}")
#     plt.axis("off")
# plt.tight_layout()
# plt.show()

b = 20 * rng.standard_normal(u_r.shape)
f = u_r + b
# ratio = np.linalg.norm(b) / (20 * np.sqrt(n))
# print(f"MSE : {mse(u_r, f):.1f} ; MAE : {mae(u_r, f):.2f} ; EMAX : {emax(u_r, f):.2f}")
# print(f"PSNR : {psnr(u_r, f):.2f} dB ; RATIO : {ratio}")

# Create noisy image
# plot_image(f, get_output_filename(SORTIES, sys.argv[1], "noisy.png"))
# A = u_r + 10
# k = int(round(100 * n / 150 ** 2))
# index = np.random.choice(n, k, replace=False)               # randomly select k pixel indices
# B = u_r.copy()
# B.flat[index] += 150
# print(f"MSE: {mse(u_r, A):.1f} ")                           # MSE = 100
# print(f"PSNR: {psnr(u_r, A):.2f} dB")
# print(f"MSE: {mse(u_r, B):.1f} ")                           # MSE = 100
# print(f"PSNR: {psnr(u_r, B):.2f} dB")
# plt.figure(figsize=(10, 2))
# plt.subplot(1, 2, 1)
# plt.axis("off")
# plt.imshow(A, cmap="gray", vmin=0, vmax=255)
# plt.title("Image with uniform noise")
# plt.subplot(1, 2, 2)
# plt.imshow(B, cmap="gray", vmin=0, vmax=255)
# plt.title("Image with noisy pixels")
# plt.axis("off")
# plt.tight_layout()
# plt.show()
# plot_image(v2, get_output_filename(SORTIES, sys.argv[1], "noisy_pixels.png"))
#=============================================================================

# Psnr = np.array([])
# lams = [0.25,0.5,1,2,4,8,16]
iterations = 200
taus = [0.2, 0.24, 0.25, 0.26, 0.3]

# for k in range(iterations):
#     print(f"iteration {k}: Energy = {energy(v, f, lam):.2f}")
#     gradient = (v - f) - lam * laplacian(v)
#     v = v - tau * gradient
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(f, cmap="gray", vmin=0, vmax=255)
# plt.title("Before: noisy image")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(v, cmap="gray", vmin=0, vmax=255)
# plt.title(f"After: {iterations} iterations")
# plt.axis("off")

#plt.show()
#================================================================================
for tau in taus:
    v = f.copy()
    for i in range(iterations):
#         print(f"iteration {i}: Energy = {energy_N_L(u_r, f):.2f}")
        v = v + tau * laplacian(v)
    print(f"tau = {tau:.2f} : max |v| = {np.max(np.abs(v)):.3g}")
#         Psnr = np.append(Psnr, psnr(u_r, v))
    if tau == 0.26:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(f, cmap="gray", vmin=0, vmax=255)
        plt.title("Before: noisy image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(v, cmap="gray", vmin=0, vmax=255)
        plt.title(f"After: {iterations} iterations with tau={tau:.2f}")
        plt.axis("off")

        plt.show()
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(f, cmap="gray", vmin=0, vmax=255)
    # plt.axis("off")
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 2)
    # plt.imshow(v, cmap="gray", vmin=0, vmax=255)
    # plt.axis("off")
# plt.plot(taus, Psnr.reshape(len(taus), iterations)[:, -1], marker='o')
# plt.xlabel('Tau')
# plt.ylabel('PSNR (dB)')
# plt.show()