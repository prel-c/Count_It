import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
from PIL import Image


def grabcut(image: np.ndarray, area_threshold: float = 0.05) -> np.ndarray | None:

    h, w, _ = image.shape
    if h < 3 or w < 3: return image
    mask = np.zeros(image.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    obj = np.zeros((1, 65), np.float64)
    rect = (1, 1, w, h)
    cv2.grabCut(image, mask, rect, background, obj, 1, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    alpha = binary.astype(np.float32)

    alpha = cv2.GaussianBlur(alpha, (11, 11), 0)

    alpha = alpha[..., np.newaxis]

    remaining_pixels = np.sum(binary)
    remaining_area_ratio = remaining_pixels / (h * w)
    """if remaining_area_ratio < area_threshold:
        return image"""

    mean_color = cv2.mean(image)[:3]
    background_cloth = np.full(image.shape, mean_color, dtype=np.uint8)
    mask_3d = binary[:, :, np.newaxis]
    segmented = (
    image.astype(np.float32) * alpha +
    background_cloth.astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)
    #segmented = cropped_img * binary[:, :, np.newaxis]

    # Uncomment to test:
    """fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(image)
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].imshow(binary, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    ax[2].imshow(segmented)
    ax[2].set_title("Segmented")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()"""
    
    return segmented

if __name__ == "__main__":

    test_image = grabcut(cv2.cvtColor(cv2.imread("D:/mygit/opd/FamNet/data/images_384_VarV2/312.jpg"), cv2.COLOR_BGR2RGB))

    if test_image.dtype != np.uint8:
        test_image = (test_image * 255).astype(np.uint8)
        test_image = test_image.astype(np.uint8)

    cv2.imwrite("D:/mygit/opd/Segmentation/image.png", test_image)
