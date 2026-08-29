
"""Final denoising settings selected during Chapter 4 calibration.
"""

import numpy as np

from .denoisers import (
    anisotropic_denoise,
    bilateral_denoise,
    non_local_means_denoise,
    total_variation_denoise,
)


# Shared noise-estimation settings
MIN_NOISE_PERCENT = 2.0
MAX_NOISE_PERCENT = 15.0
FLAT_FRACTION = 0.04

METHOD_ORDER = ["AD", "TV", "BF", "NLM"]
METHOD_LABELS = {
    "AD": "Anisotropic diffusion",
    "TV": "Total variation",
    "BF": "Bilateral filtering",
    "NLM": "Non-local means",
}
PRIMARY_PARAMETER_NAMES = {
    "AD": "kappa",
    "TV": "weight",
    "BF": "sigma_color",
    "NLM": "h",
}
SELECTED_MAPPINGS = {
    "AD": "fitted quadratic",
    "TV": "piecewise-linear interpolation",
    "BF": "piecewise-linear interpolation",
    "NLM": "piecewise-linear interpolation",
}


# Calibration points used by the final mappings
CALIBRATED_NOISE_LEVELS = np.array(
    [2, 4, 6, 8, 10, 12, 15], dtype=float
)

AD_K_VALUES = np.array(
    [0.02, 0.04, 0.07, 0.10, 0.13, 0.17, 0.23], dtype=float
)
TV_WEIGHT_VALUES = np.array(
    [0.0100, 0.0225, 0.0400, 0.0575, 0.0750, 0.0950, 0.1325],
    dtype=float,
)
BF_SIGMA_COLOR_VALUES = np.array(
    [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.22], dtype=float
)
NLM_H_VALUES = np.array(
    [0.0225, 0.0425, 0.0625, 0.0825, 0.1025, 0.1250, 0.1600],
    dtype=float,
)


# AD uses the fitted quadratic selected during validation
AD_K_COEFFICIENTS = np.array(
    [0.0004241321823822198, 0.009033064730236873, -0.0006712207209469963]
)


# Fixed secondary settings
AD_FIXED_SETTINGS = {
    "n_iter": 11,
    "gamma": 0.10,
    "conduction": "reciprocal",
}
BF_FIXED_SETTINGS = {"sigma_spatial": 2}
NLM_FIXED_SETTINGS = {"patch_size": 5, "patch_distance": 3}


# Add the four fixed-baseline primary parameters after calibration
FIXED_PRIMARY_PARAMETERS = {
    "AD": 0.1300,
    "TV": 0.0675,
    "BF": 0.1000,
    "NLM": 0.1050,
}


def normalise_method(method):
    method = str(method).upper()
    if method not in METHOD_ORDER:
        raise ValueError("method must be 'AD', 'TV', 'BF', or 'NLM'")
    return method


def clip_noise_percent(noise_percent):
    """Restrict a noise estimate to the calibrated range."""
    return float(np.clip(
        noise_percent,
        MIN_NOISE_PERCENT,
        MAX_NOISE_PERCENT,
    ))


def select_mapped_parameter(method, noise_percent):
    """Return the mapped primary parameter for one denoising method."""
    method = normalise_method(method)
    noise_percent = clip_noise_percent(noise_percent)

    if method == "AD":
        value = np.polyval(AD_K_COEFFICIENTS, noise_percent)
        return float(np.clip(value, AD_K_VALUES.min(), AD_K_VALUES.max()))

    if method == "TV":
        return float(np.interp(
            noise_percent,
            CALIBRATED_NOISE_LEVELS,
            TV_WEIGHT_VALUES,
        ))

    if method == "BF":
        return float(np.interp(
            noise_percent,
            CALIBRATED_NOISE_LEVELS,
            BF_SIGMA_COLOR_VALUES,
        ))

    return float(np.interp(
        noise_percent,
        CALIBRATED_NOISE_LEVELS,
        NLM_H_VALUES,
    ))


def get_mapped_parameters(method, noise_percent):
    """Return all denoiser arguments for a mapped noise value."""
    method = normalise_method(method)
    value = select_mapped_parameter(method, noise_percent)

    if method == "AD":
        return {
            "n_iter": AD_FIXED_SETTINGS["n_iter"],
            "kappa": value,
            "gamma": AD_FIXED_SETTINGS["gamma"],
            "conduction": AD_FIXED_SETTINGS["conduction"],
        }
    if method == "TV":
        return {"weight": value}
    if method == "BF":
        return {
            "sigma_color": value,
            "sigma_spatial": BF_FIXED_SETTINGS["sigma_spatial"],
        }
    return {
        "h": value,
        "patch_size": NLM_FIXED_SETTINGS["patch_size"],
        "patch_distance": NLM_FIXED_SETTINGS["patch_distance"],
    }


def get_fixed_parameters(method):
    """Return all denoiser arguments for the fixed baseline."""
    method = normalise_method(method)
    value = FIXED_PRIMARY_PARAMETERS[method]

    if value is None:
        raise ValueError(
            f"Set FIXED_PRIMARY_PARAMETERS['{method}'] from calibration data "
            "before running the fixed baseline."
        )

    if method == "AD":
        parameters = AD_FIXED_SETTINGS.copy()
        parameters["kappa"] = value
        return parameters
    if method == "TV":
        return {"weight": value}
    if method == "BF":
        parameters = BF_FIXED_SETTINGS.copy()
        parameters["sigma_color"] = value
        return parameters

    parameters = NLM_FIXED_SETTINGS.copy()
    parameters["h"] = value
    return parameters


def denoise_with_parameters(image, method, parameters):
    """Apply one denoiser with an explicit parameter dictionary."""
    method = normalise_method(method)

    if method == "AD":
        denoised_image, _ = anisotropic_denoise(image, **parameters)
        return denoised_image
    if method == "TV":
        return total_variation_denoise(image, **parameters)
    if method == "BF":
        return bilateral_denoise(image, **parameters)
    return non_local_means_denoise(image, **parameters)


def denoise_mapped(image, method, noise_percent):
    """Apply the final mapping using an estimated or known noise value."""
    parameters = get_mapped_parameters(method, noise_percent)
    return denoise_with_parameters(image, method, parameters)


def denoise_fixed(image, method):
    """Apply the fixed baseline after its primary values have been set."""
    parameters = get_fixed_parameters(method)
    return denoise_with_parameters(image, method, parameters)
