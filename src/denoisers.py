import numpy as np
from skimage.restoration import denoise_bilateral


def anisotropic_denoise(image, n_iter, kappa, gamma,
                        conduction="exponential"):
    # Convert the image to float
    denoised_image = image.astype(np.float32).copy()

    for _ in range(n_iter):
        # Calculate differences with four neighbouring pixels
        north = np.zeros_like(denoised_image)
        south = np.zeros_like(denoised_image)
        east = np.zeros_like(denoised_image)
        west = np.zeros_like(denoised_image)

        north[:-1, :] = denoised_image[1:, :] - denoised_image[:-1, :]
        south[1:, :] = denoised_image[:-1, :] - denoised_image[1:, :]
        east[:, :-1] = denoised_image[:, 1:] - denoised_image[:, :-1]
        west[:, 1:] = denoised_image[:, :-1] - denoised_image[:, 1:]

        # Calculate conduction coefficients
        if conduction == "exponential":
            c_north = np.exp(-(north / kappa) ** 2)
            c_south = np.exp(-(south / kappa) ** 2)
            c_east = np.exp(-(east / kappa) ** 2)
            c_west = np.exp(-(west / kappa) ** 2)

        elif conduction == "reciprocal":
            c_north = 1 / (1 + (north / kappa) ** 2)
            c_south = 1 / (1 + (south / kappa) ** 2)
            c_east = 1 / (1 + (east / kappa) ** 2)
            c_west = 1 / (1 + (west / kappa) ** 2)

        else:
            raise ValueError(
                "conduction must be 'exponential' or 'reciprocal'"
            )

        # Update the image
        denoised_image += gamma * (
            c_north * north
            + c_south * south
            + c_east * east
            + c_west * west
        )

    return np.clip(denoised_image, 0, 1)


def bilateral_denoise(image, sigma_color, sigma_spatial):
    # Apply bilateral filtering
    denoised_image = denoise_bilateral(
        image,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=None
    )

    return denoised_image