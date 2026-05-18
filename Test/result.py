import cv2
import numpy as np

from test import test
#7715
epsilon=[]
ep1=[]
S=[]
for i in range(2, 7716):
    try:
        with open(f"/mnt/c/CV/FSC147_modifications/FSC147B4/yolo_labels/{i}.txt", 'r') as f:
            A = []
            for lines in f:
                # Парсим строку: class_id x_center y_center width height
                parts = lines.strip().split()
                # Берем только координаты (x_center, y_center, width, height)
                coords = [float(x) for x in parts[1:5]]
                A.append(coords)

            image_RGB = cv2.imread(f'/mnt/c/CV/FSC147_modifications/FSC147B4/images_384_VarV2/{i}.jpg')
            image=cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
            image1=np.copy(image)
            height, width = image.shape

            positions=np.array(A)
            # Сначала преобразуем абсолютные координаты
            x_center = positions[:, 0] * width
            y_center = positions[:, 1] * height
            box_width = positions[:, 2] * width
            box_height = positions[:, 3] * height

            # Вычисляем углы
            x1 = (x_center - box_width/2).astype(np.int64)
            y1 = (y_center - box_height/2).astype(np.int64)
            x2 = (x_center + box_width/2).astype(np.int64)
            y2 = (y_center + box_height/2).astype(np.int64)

            # Клиппинг координат (обрезаем по границам)
            x1 = np.clip(x1, 0, width-1)
            y1 = np.clip(y1, 0, height-1)
            x2 = np.clip(x2, 1, width)
            y2 = np.clip(y2, 1, height)

            # Собираем обратно
            positions = np.stack([x1, y1, x2, y2], axis=1)

            # Для шаблона берем второй объект
            x1, y1, x2, y2 = positions[1]
            template = image[y1:y2, x1:x2]
        res, image1=test(image, template)
        cv2.imwrite(f"/mnt/c/CV/Test/results/{i}.jpg", image1)
        ep=abs(len(positions)-res)/len(positions)
        epsilon.append(ep)
        ep1.append((ep, i))
        print(i)
        if ep>0.4:
            S.append(i)
    except FileNotFoundError:                                            #(FileNotFoundError, cv2.error)
        continue
print(ep1)
print(sum(epsilon)/len(epsilon))
print(S)