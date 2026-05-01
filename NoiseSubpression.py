import numpy as np
import cv2
from GrabCutSegmentation import grabcut


def no_noise(density_map: np.ndarray, sq: int) -> float:
    kernel_size = (int(np.sqrt(sq)), int(np.sqrt(sq)))
    kernel = np.ones(kernel_size, np.uint8)
    map_no_noise = cv2.morphologyEx(density_map, cv2.MORPH_OPEN, kernel)
    return np.sum(map_no_noise)


if __name__ == '__main__':
    density = np.load('density_map_result.npy').squeeze()
    image = cv2.imread('materials-python-pillow/birds.jpg')
    segment, square = grabcut(image)
    print(no_noise(density, square))
