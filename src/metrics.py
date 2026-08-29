import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# Kernels used by GMSD
average_kernel = np.full((2, 2), 0.25, dtype=np.float64)
prewitt_x_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float64) / 3
prewitt_y_kernel = prewitt_x_kernel.T

# Equivalent to 170 on the 0-255 intensity scale
gmsd_stability_constant = 170 / (255 ** 2)


def _downsample_image(image):
    smoothed_image = convolve2d(
        image, average_kernel, mode="same", boundary="symm"
    )
    return smoothed_image[::2, ::2]


def _gradient_magnitude(image):
    gradient_x = convolve2d(
        image, prewitt_x_kernel, mode="same", boundary="symm"
    )
    gradient_y = convolve2d(
        image, prewitt_y_kernel, mode="same", boundary="symm"
    )
    return np.hypot(gradient_x, gradient_y)


def calculate_psnr(clean_image, test_image):
    # Calculate PSNR
    return peak_signal_noise_ratio(
        clean_image,
        test_image,
        data_range=1.0
    )


def calculate_ssim(clean_image, test_image):
    # Calculate SSIM
    return structural_similarity(
        clean_image,
        test_image,
        data_range=1.0
    )


def calculate_gmsd(clean_image, test_image):
    """Calculate GMSD for two grayscale images on the [0, 1] scale."""
    clean_image = np.asarray(clean_image, dtype=np.float64)
    test_image = np.asarray(test_image, dtype=np.float64)

    if clean_image.shape != test_image.shape:
        raise ValueError("The two images must have the same shape")
    if clean_image.ndim != 2:
        raise ValueError("GMSD requires two grayscale images")
    if min(clean_image.shape) < 2:
        raise ValueError("The images are too small for GMSD")
    if not np.isfinite(clean_image).all() or not np.isfinite(test_image).all():
        raise ValueError("The images contain a non-finite value")

    # Downsample before calculating the gradient magnitudes
    clean_gradient = _gradient_magnitude(_downsample_image(clean_image))
    test_gradient = _gradient_magnitude(_downsample_image(test_image))

    similarity_map = (
        2 * clean_gradient * test_gradient + gmsd_stability_constant
    ) / (
        clean_gradient ** 2 + test_gradient ** 2
        + gmsd_stability_constant
    )

    # Lower GMSD indicates less gradient distortion
    return float(np.std(similarity_map))
