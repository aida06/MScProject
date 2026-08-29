import os

import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float, img_as_ubyte


def load_grayscale_image(path):
    # Check whether the image exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Read the image as grayscale
    image = io.imread(path, as_gray=True)

    # Convert pixel values to the range [0, 1]
    return img_as_float(image)


def save_image(path, image):
    # Create the output folder if needed
    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    # Convert the image before saving
    io.imsave(path, img_as_ubyte(image))


def show_images(images, titles, save_path=None):
    # Create one row of images
    plt.figure(figsize=(4 * len(images), 4))

    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        plt.title(titles[i])
        plt.axis("off")
        # plt.colorbar(label="", orientation="vertical")

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        folder = os.path.dirname(save_path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()