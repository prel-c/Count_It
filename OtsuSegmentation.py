import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
import cv2


def otsu(filename: str, sigma: float) -> np.ndarray | None:
    def crop(image_name: str) -> np.ndarray | None:
        img2crop = cv2.imread(image_name)
        if img2crop is None:
            return None
        roi = cv2.selectROI("Select Area", img2crop, fromCenter=False, showCrosshair=True)
        x, y, w, h = map(int, roi)
        cv2.destroyAllWindows()
        cropped_img = img2crop[y:y + h, x:x + w]
        return cropped_img

    image = crop(filename).astype(np.float64) / 255.0
    image_gray = sk.color.rgb2gray(image)
    image_blurred = sk.filters.gaussian(image_gray, sigma=sigma)
    thresh = sk.filters.threshold_otsu(image_blurred)
    binary = image_blurred > thresh
    mask = np.expand_dims(binary, 2)
    segmented = image * mask
    # Uncomment to test:
    """
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(image.astype(np.float64))
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].imshow(binary.astype(np.float64), cmap="gray")
    ax[1].set_title("Otsu")
    ax[1].axis("off")

    ax[2].imshow(segmented.astype(np.float64))
    ax[2].set_title("Segmented")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()
    """
    return segmented


test_image = otsu("your_image.jpg", 1.5)
