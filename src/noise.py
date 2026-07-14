import numpy as np


def add_gaussian_noise(image, sigma, seed=42):
    # Use the same random noise each time
    np.random.seed(seed)

    # Create Gaussian noise
    noise = np.random.normal(0, sigma / 255.0, image.shape)

    # Add noise to the image
    noisy_image = image + noise

    # Keep pixel values between 0 and 1
    return np.clip(noisy_image, 0, 1)