import sys
import matplotlib.pyplot as plt
from matplotlib.style import use
from skimage import io, color
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from Tp1_Math_Functions import A, AT, D, DT, power_method_laplacian, read_image, resize_image, to_float, to_grayscale 

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

norm_A, estimates_A = power_method_laplacian(A, AT, f.shape, 200)

norm_D, estimates_D = power_method_laplacian(D, DT, f.shape, 200)

print("Norm A:", norm_A)
print("Norm D:", norm_D)
ratio_A = np.linalg.norm(A(b)) / np.linalg.norm(b)

print("Estimated norm A:", norm_A)
print("Number of iterations:", len(estimates_A))
print("Observed amplification:", ratio_A)

import matplotlib.pyplot as plt

plt.plot(estimates_A, label="Averaging operator A")
plt.plot(estimates_D, label="Gradient operator D")

plt.axhline(1, linestyle="--", label="Theoretical norm A = 1")
plt.axhline(np.sqrt(8), linestyle="--", label="Theoretical norm D = sqrt(8)")

plt.xlabel("Iteration")
plt.ylabel("Estimated operator norm")
plt.title("Convergence of the Power Method")
plt.legend()
plt.grid()

plt.show()