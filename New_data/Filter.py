import os
import cv2

for i in range (8000):
    try:
        with open(f"annotations/{i}.txt") as file:
            data=file.read()
            image=cv2.imread(f"images_384_VarV2/{i}.jpg")
    except FileNotFoundError:
        try:
            os.remove(f"images_384_VarV2/{i}.jpg")
            print(f"файл {i}.jpg удалён")
        except FileNotFoundError:
            continue
        continue