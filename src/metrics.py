from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


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