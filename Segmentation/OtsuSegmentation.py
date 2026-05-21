import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
import cv2


def otsu(image: np.ndarray, sigma: float) -> np.ndarray | None:

    image = image.astype(np.float64) / 255.0
    image_gray = sk.color.rgb2gray(image)
    image_blurred = sk.filters.gaussian(image_gray, sigma=sigma)
    thresh = sk.filters.threshold_otsu(image_blurred)
    binary = image_blurred > thresh

    mask = (~binary).astype(np.float32)

    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    mask = np.expand_dims(mask, axis=2)

    segmented = image * mask
    segmented = np.clip(segmented * 255, 0, 255).astype(np.uint8)
    #segmented = image * (~binary)[:, :, np.newaxis]
    # Uncomment to test:

    """fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(image)
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].imshow(binary, cmap="gray")
    ax[1].set_title("Otsu")
    ax[1].axis("off")

    ax[2].imshow(segmented)
    ax[2].set_title("Segmented")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()"""

    return segmented

if __name__ == "__main__":

    test_image = otsu(cv2.imread("D:/mygit/opd/FamNet/data/images_384_VarV2/312.jpg"), 2)

    cv2.imwrite("image.png", test_image)
