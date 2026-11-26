# this code is used to test the Averageur function (which takes the average of the neighboring pixels) 
# and compare it with the median3x3 function (which takes the median of the neighboring pixels)
# on a synthetic image with noise.
import sys
import matplotlib.pyplot as plt
from matplotlib.style import use
from skimage import io, color
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from Tp_Mine.Tp1_Math_Functions import energy_N_L, read_image, to_grayscale, median3x3, Averageur
from Tp_Mine.Tp1_Math_Functions import to_float, get_output_filename, plot_image, resize_image
from Tp_Mine.Tp1_Error_Functions import mse, mae, emax, psnr, laplacian

u = read_image(sys.argv[1])
u = to_grayscale(u)
u = to_float(u)

# Resized image 512x512
u_r = resize_image(u, 512, 512)

# rows = 200
# cols = 200
# u_r = np.zeros((rows, cols))

# for i in range(rows):
#     for j in range(cols):
#         if (i < rows //2 and j < cols // 2) or (i >= rows // 2 and j >= cols // 2):
#             u_r[i, j] = 1 

# Gaussian noise
rng = np.random.default_rng(0)
mask = rng.random(u_r.shape) < 0.05
u_r[mask] = 1

Med = median3x3(u_r)
M = Averageur(u_r)

# plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(u_r, cmap="gray", vmin=u_r.min(), vmax=u_r.max())
# plt.title("Median 3x3")
plt.subplot(1, 3, 2)
plt.imshow(M, cmap="gray", vmin=M.min(), vmax=M.max())
plt.subplot(1, 3, 3)
plt.imshow(Med, cmap="gray", vmin=Med.min(), vmax=Med.max())
# plt.title("Averageur")
plt.axis("off")
plt.show()