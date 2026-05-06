import cv2
import numpy as np
import pandas as pd
from test import test

for i in range(2, 7715):
    try:
        image_RGB = cv2.imread(f"images_384_VarV2/{i}.jpg")
        image=cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
        A=[]
        with open(f"yolo_labels/{i}.txt", 'r') as f:
            for line in f:
                a=[float(x) for x in line[2:-2].split(" ")]
                A.append(a)
        pos=A[0]
        res=test(image, image[A[0][1]:A[0][1]+A[0][3]][A[0][0]:A[0][0]+A[0][2]])
    except FileNotFoundError:
        continue 
print(res)

