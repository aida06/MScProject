import numpy as np
import pywt
from scipy.signal import convolve2d


def laplacian_noise_estimate(image):
    # Noise-estimation kernel
    kernel = np.array([
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]
    ])

    # Apply the kernel without border padding
    response = convolve2d(
        image,
        kernel,
        mode="valid"
    )

    height, width = image.shape

    # Estimate noise on the 0-1 scale
    sigma = (
        np.sqrt(np.pi / 2)
        * np.sum(np.abs(response))
        / (6 * (width - 2) * (height - 2))
    )

    return sigma


def edge_aware_laplacian_noise_estimate(image, flat_fraction):

    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")

    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("image must be at least 3 x 3 pixels")

    if not np.all(np.isfinite(image)):
        raise ValueError("image contains non-finite values")

    if not 0 < flat_fraction <= 1:
        raise ValueError("flat_fraction must be in the interval (0, 1]")

    laplacian_kernel = np.array([
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]
    ], dtype=np.float64)

    gradient_x_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    gradient_y_kernel = gradient_x_kernel.T

    laplacian_response = convolve2d(
        image,
        laplacian_kernel,
        mode="valid"
    )
    gradient_x = convolve2d(
        image,
        gradient_x_kernel,
        mode="valid"
    )
    gradient_y = convolve2d(
        image,
        gradient_y_kernel,
        mode="valid"
    )

    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_threshold = np.quantile(
        gradient_magnitude,
        flat_fraction
    )
    flat_mask = gradient_magnitude <= gradient_threshold
    selected_response = laplacian_response[flat_mask]

    if selected_response.size == 0:
        return float(laplacian_noise_estimate(image))

    sigma = (
        np.sqrt(np.pi / 2)
        * np.mean(np.abs(selected_response))
        / 6
    )

    return float(sigma)


def wavelet_noise_estimate(image):
    # Apply a one-level wavelet transform
    _, (_, _, diagonal) = pywt.dwt2(
        image,
        "db1"
    )

    # Estimate noise on the 0-1 scale
    sigma = (np.median(np.abs(diagonal))/ 0.6745)

    return sigma



def block_noise_estimate(image, return_details=False):
    # Positions used by the 12 directional masks
    directions = [
        [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
        [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)],
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2)],
        [(2, 0), (2, 1), (2, 2), (3, 2), (4, 2)],
        [(4, 2), (3, 2), (2, 2), (2, 3), (2, 4)],
        [(0, 2), (1, 2), (2, 2), (2, 3), (2, 4)],
        [(1, 0), (0, 1), (0, 2), (0, 3), (1, 4)],
        [(3, 0), (4, 1), (4, 2), (4, 3), (3, 4)],
        [(0, 1), (1, 0), (2, 0), (3, 0), (4, 1)],
        [(0, 3), (1, 4), (2, 4), (3, 4), (4, 3)]
    ]
    coefficients = np.array([-1, -1, 4, -1, -1])

    # Crop the image into complete 5 × 5 blocks
    height = image.shape[0] - image.shape[0] % 5
    width = image.shape[1] - image.shape[1] % 5
    cropped = image[:height, :width]

    block_variances = []
    homogeneity_values = []
    block_positions = []

    for row in range(0, height, 5):
        for col in range(0, width, 5):
            block = cropped[row:row + 5, col:col + 5]
            homogeneity = 0

            # Combine responses from all directions
            for direction in directions:
                values = np.array([
                    block[r, c]
                    for r, c in direction
                ])
                homogeneity += abs(
                    np.sum(coefficients * values)
                )

            block_variances.append(np.var(block))
            homogeneity_values.append(homogeneity)
            block_positions.append((row, col))

    block_variances = np.array(block_variances)
    homogeneity_values = np.array(homogeneity_values)

    # Use the three most homogeneous blocks as a reference
    reference_indices = np.argsort(
        homogeneity_values
    )[:3]
    reference_variance = np.median(
        block_variances[reference_indices]
    )

    # Select blocks with similar variances
    minimum_variance = np.min(block_variances)
    threshold = abs(
        reference_variance - minimum_variance
    )
    selected_indices = np.where(
        np.abs(block_variances - reference_variance)
        <= threshold
    )[0]

    if len(selected_indices) == 0:
        selected_indices = reference_indices

    # Average the selected block variances
    noise_variance = np.mean(
        block_variances[selected_indices]
    )
    sigma = np.sqrt(noise_variance)

    if return_details:
        selected_positions = [
            block_positions[index]
            for index in selected_indices
        ]
        return sigma, selected_positions, cropped

    return sigma
