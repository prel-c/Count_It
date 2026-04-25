import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from utils import matlab_style_gauss2D

def generate_density_map(img_shape, points, sigma=3):
    'Генерирует "идеальную" карту плотности для обучения'
    density_map = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
    h, w = img_shape[0], img_shape[1]

    for x, y, w1, h1 in points:
        x, y = int(round(x)), int(round(y))
        w1, h1 = int(round(w1)), int(round(h1))
        kernel = int(min(w1, h1) * 0.8)
        sigma = kernel // 5
        if kernel % 2 == 0: kernel -= 1
        pad = kernel // 2
        # Пропускаем точки, которые оказались вне картинки
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + pad + 1)
        y2 = min(h, y + pad + 1)

        kx1 = pad - (x - x1)
        ky1 = pad - (y - y1)
        kx2 = pad + (x2 - x)
        ky2 = pad + (y2 - y)

        density_map[y1:y2, x1:x2] += matlab_style_gauss2D(shape=(kernel, kernel), sigma=sigma)[ky1:ky2, kx1:kx2]
    
    total_points = len(points)
    if total_points > 0 and density_map.sum() != 0:
        density_map = density_map / density_map.sum() * total_points

    return density_map

def yolo_to_pixels(bbox, img_width, img_height):
    class_id, center_x, center_y, width, height = bbox
    
    x_center = center_x * img_width
    y_center = center_y * img_height
    box_width = width * img_width
    box_height = height * img_height
    
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)
    
    return (x1, y1, x2, y2)
path = "data/my/Pipes_340"
image = cv.imread(path + "/images/pipes_pipes_images (48)_1__1.jpg")

h, w, c = image.shape
points_of_center = []

with open(path + "/labels/pipes_pipes_images (48)_1__1.txt", 'r', encoding='utf-8') as data:
    for box in data:
        box = tuple(map(float, box.split()))
        points_of_center.append((box[1]*w, box[2]*h, box[3]*w, box[4]*h))
        x1, y1, x2, y2 = yolo_to_pixels(box, w, h)
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

file = np.load("density_map_result.npy")
file = np.squeeze(file)
plt.xlabel(sum(sum(file)))
plt.imshow(file)
plt.savefig("output.png")
