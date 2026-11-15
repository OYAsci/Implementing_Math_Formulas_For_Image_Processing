import sys
import matplotlib.pyplot as plt
from matplotlib.style import use
from skimage import io, color
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from Tp1_Math_Functions import energy_N_L, read_image, to_grayscale, energy 
from Tp1_Math_Functions import to_float, get_output_filename, plot_image, resize_image
from Tp1_Error_Functions import mse, mae, emax, psnr, laplacian

u = read_image(sys.argv[1])
u = to_grayscale(u)
u = to_float(u)

# Resized image 512x512
u_r = resize_image(u, 512, 512)
n = u_r.size

# Generate Gaussian noise
rng = np.random.default_rng(0)
b = 20 * rng.standard_normal(u_r.shape)
f = u_r + b

Psnr = np.array([])
lam = 1
iterations = 50
taus = [0.1, 0.14, 0.15, 0.16, 0.2]
for tau in taus:
    v = f.copy()
    L = 1 + 8 * lam
    for k in range(iterations):
        print(f"tau={tau:.2f}, iteration {k}: Energy = {energy(v, f, lam):.2f}")
        gradient = (v - f) - lam * laplacian(v)
        v = v - tau * gradient
        Psnr = np.append(Psnr, psnr(u_r, v))
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
