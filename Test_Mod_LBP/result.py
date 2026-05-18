import cv2
import numpy as np

from test import test
#7715
epsilon=[]
S=[]
for i in range(1, 7716):
    try:
        with open(f"/mnt/c/CV/FSC147_modifications/FSC147B5/yolo_labels/{i}.txt", 'r') as f:
            A = []
            for lines in f:
                # Парсим строку: class_id x_center y_center width height
                parts = lines.strip().split()
                # Берем только координаты (x_center, y_center, width, height)
                coords = [float(x) for x in parts[1:5]]
                A.append(coords)

            image_RGB = cv2.imread(f'/mnt/c/CV/FSC147_modifications/FSC147B5/images_384_VarV2/{i}.jpg')
            image=cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
            image1=np.copy(image)
            height, width = image.shape

            positions=np.array(A)
            positions[:, [0]]-=positions[:, [2]]/2
            positions[:, [1]]-=positions[:, [3]]/2
            positions[:, [2]] +=positions[:, [0]]
            positions[:, [3]] +=positions[:, [1]]
            

            positions[:, [0, 2]] *= width
            positions[:, [1, 3]] *= height
            positions=np.int64(positions)

            for h in range (len (positions)):
                cv2.rectangle(image1, (positions[h, 0], positions[h, 1]), (positions[h, 2], positions[h, 3]), (0, 255, 0), 2)
             
            template=image[positions[1, 1]:positions[1, 3], positions[1, 0]:positions[1, 2]]

        res=test(image, template)
        ep=abs(len(positions)-res)/len(positions)
        epsilon.append(ep)
        print(i)
        if ep>0.5:
            S.append(i)
    except (FileNotFoundError, cv2.error):
        continue
#print(epsilon)
print(sum(epsilon)/len(epsilon))
#print(S)