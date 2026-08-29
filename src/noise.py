import numpy as np


def add_gaussian_noise(image, sigma, seed=42):
    # Generate Gaussian noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)

    # Add noise without clipping
    return image + noise