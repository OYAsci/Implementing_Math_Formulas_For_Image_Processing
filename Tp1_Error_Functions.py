import numpy as np

# Error
def Error(u, v):
    """Calculates the error between two images u and v."""
    e = u.astype(float) - v.astype(float)
    return e

# Mean Squared Error
def mse(u, v):
    return np.mean(Error(u,v) ** 2)

# Mean Absolute Error
def mae(u, v):
    return np.mean(np.abs(Error(u,v)))

# Maximum Absolute Error
def emax(u, v):
    return np.max(np.abs(Error(u,v)))

# Peak Signal-to-Noise Ratio
def psnr(u, v):
    error = mse(u, v)

    if error == 0:
        return float("inf")

    return 10 * np.log10(255 ** 2 / error)

# laplacian
def laplacian(v):
    p = np.pad(v, 1, mode="edge")

    return (
        p[:-2, 1:-1]   # top
        + p[2:, 1:-1]  # bottom
        + p[1:-1, :-2] # left
        + p[1:-1, 2:]  # right
        - 4 * v
    )