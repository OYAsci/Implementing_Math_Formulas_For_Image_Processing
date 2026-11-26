import os
import sys
from skimage.transform import resize 
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, data

# Create output directory
def create_output_directory():
    ICI = os.path.dirname(os.path.abspath(__file__))
    SORTIES = os.path.join(ICI, "sorties")
    os.makedirs(SORTIES, exist_ok=True)

    return SORTIES

# Load image
def read_image(path):
    u = io.imread(path)
    return u

#Convert to grayscale
def to_grayscale(u):
    if u.ndim == 3:
        u = 255 * color.rgb2gray(u[..., :3])

    return u

#Convert pixel values to float
def to_float(u):
    u = u.astype(float)
    return u

# Get output filename
def get_output_filename(SORTIES, input_path, output_name=None):
    if output_name is not None:
        filename = output_name
    else:
        original_name = os.path.basename(input_path)
        name = os.path.splitext(original_name)[0]
        filename = name + "_Sor.png"

    name, extension = os.path.splitext(filename)

    output_path = os.path.join(SORTIES, filename)

    counter = 1

    while os.path.exists(output_path):
        new_filename = f"{name}_{counter}{extension}"
        output_path = os.path.join(SORTIES, new_filename)
        counter += 1

    return output_path

# Create figure
def plot_image(u, filename):
    fig, ax = plt.subplots()

    if u.ndim == 2:
        ax.imshow(u, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(u)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    
# resize image
def resize_image(u, height, width):
    return resize(u, (height, width), preserve_range=True) 

# Compute the energy of an image
def energy(v, f, lam):
    E1 = 0.5 * np.sum((v - f)**2)

    dx = v[:, 1:] - v[:, :-1]
    dy = v[1:, :] - v[:-1, :]

    E2 = (lam / 2) * (np.sum(dx**2) + np.sum(dy**2))

    return E1 + E2

# compute the energy of an image without lambda
def energy_N_L(v, f):
    E1 = 0.5 * np.sum((v - f)**2)

    dx = v[:, 1:] - v[:, :-1]
    dy = v[1:, :] - v[:-1, :]

    E2 = 0.5 * (np.sum(dx**2) + np.sum(dy**2))

    return E1 + E2

# power method to compute the largest eigenvalue of the Laplacian operator
def power_method_laplacian(A, AT, shape, num_iterations):
    x = np.random.rand(*shape)
    x /= np.linalg.norm(x)

    estimates = []

    for _ in range(num_iterations):
        y = AT(A(x))
        norm_y = np.linalg.norm(y)

        if norm_y == 0:
            break

        estimate = np.sqrt(norm_y)
        estimates.append(estimate)

        x = y / norm_y

    if not estimates:
        return 0.0, estimates

    return estimates[-1], estimates

# 3x3 averaging filter
def A(x):
    result = np.zeros_like(x)

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            result += np.roll(np.roll(x, i, axis=0), j, axis=1)

    return result / 9


# Adjoint of A
def AT(x):
    return A(x)


# Discrete gradient
def D(x):
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x

    return np.stack((dx, dy), axis=0)


# Adjoint of D
def DT(y):
    dx = y[0]
    dy = y[1]

    return (
        np.roll(dx, 1, axis=1) - dx
        + np.roll(dy, 1, axis=0) - dy
    )

def Averageur(u):
    """
    Compute the average of a 2D array u.
    """
    u = np.asarray(u, dtype=float)
    v = np.zeros_like(u)

    for a in range(-1, 2):
        for b in range(-1, 2):
            shifted = np.roll(u, shift=a, axis=0)
            shifted = np.roll(shifted, shift=b, axis=1)
            v += shifted

    return v / 9

def median3x3(u):
    u = np.asarray(u, dtype=float)
    neighbours = []

    for a in range(-1, 2):
        for b in range(-1, 2):
            shifted = np.roll(u, shift=a, axis=0)
            shifted = np.roll(shifted, shift=b, axis=1)
            neighbours.append(shifted)

    neighbours = np.stack(neighbours, axis=0)

    return np.median(neighbours, axis=0)