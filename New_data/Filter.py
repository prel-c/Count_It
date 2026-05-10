import pandas as pd
import numpy as np
import cv2
#7715
for i in range(1, 3):
    try:
        with open(f"yolo_labels/{i}.txt", 'r') as f:
            A = []
            for lines in f:
                # Парсим строку: class_id x_center y_center width height
                parts = lines.strip().split()
                # Берем только координаты (x_center, y_center, width, height)
                coords = [float(x) for x in parts[1:5]]
                A.append(coords)
            

            image_RGB = cv2.imread(f"images_384_VarV2/{i}.jpg")
            image=cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
            image1=np.copy(image)
            height, width = image.shape


            positions=np.array(A)
            sizes=np.array(())
            sizes[:]=positions[:, [2]]*positions[:, [3]]*width*height

            positions[:, [0]]-=positions[:, [2]]/2
            positions[:, [1]]-=positions[:, [3]]/2
            positions[:, [2]] +=positions[:, [0]]
            positions[:, [3]] +=positions[:, [1]]
            positions[:, [0, 2]] *= width
            positions[:, [1, 3]] *= height
            positions=np.int64(positions)

            print(positions)
    except FileNotFoundError:
        continue