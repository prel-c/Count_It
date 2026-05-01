import cv2
import numpy as np
import matplotlib.pyplot as plt


def grabcut(image: np.ndarray ) -> np.ndarray | None:
    """img = cv2.imread(image_name)
    if img is None:
        return None
    window_name = "Select Area"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        return None
    cropped_img = img[y:y + h, x:x + w]"""      

    cropped_img = image
    h, w, _ = cropped_img.shape
    mask = np.zeros(cropped_img.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    obj = np.zeros((1, 65), np.float64)
    rect = (1, 1, w - 2, h - 2)
    cv2.grabCut(cropped_img, mask, rect, background, obj, 3, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = cropped_img * binary[:, :, np.newaxis]

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

    test_image = grabcut(cv2.cvtColor(cv2.imread("D:/mygit/opd/FamNet/data/images_384_VarV2/5.jpg"), cv2.COLOR_BGR2RGB))

    if test_image.dtype != np.uint8:
        test_image = (test_image * 255).astype(np.uint8)
        test_image = test_image.astype(np.uint8)

    cv2.imwrite("D:/mygit/opd/Segmentation/image.png", test_image)
