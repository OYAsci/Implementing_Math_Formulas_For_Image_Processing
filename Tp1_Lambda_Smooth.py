import sys
import matplotlib.pyplot as plt
from matplotlib.style import use
from skimage import io, color
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from Tp1 import energy_N_L, read_image, to_grayscale, energy 
from Tp1 import to_float, get_output_filename, plot_image, resize_image
from Errors import mse, mae, emax, psnr, laplacian

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
lams = [0.25,0.5,1,2,4,8,16]
iterations = 50
for lam in lams:
    v = f.copy()
    L = 1 + 8 * lam
    tau = 2 / L
    for k in range(iterations):
        print(f"lam={lam:.2f}, iteration {k}: Energy = {energy(v, f, lam):.2f}")
        gradient = (v - f) - lam * laplacian(v)
        v = v - tau * gradient
        Psnr = np.append(Psnr, psnr(u_r, v))
plt.plot(lams, Psnr.reshape(len(lams), iterations)[:, -1], marker='o')
plt.xlabel('Lambda')
plt.ylabel('PSNR (dB)')
plt.show()