import pandas as pd
import numpy as np
import cv2
import os
import json

with open("annotation_FSC147_384.json", "r", encoding="utf-8") as file:
    data=json.load(file)

for i in range(1, 8000):
    try:
        a=data[f"{i}.jpg"]
    except KeyError:
        try:
            os.remove(f"images_384_VarV2/{i}.jpg")
            print(f"файл {i}.jpg удалён")
        except FileNotFoundError:
            continue
