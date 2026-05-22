import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import warnings
warnings.filterwarnings("ignore")
import gc
import copy
from PIL import Image
import cv2
import json
import torch
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
from easydict import EasyDict

#from YOLO.yolopipe import *

def get_box(image, num_box=None):
    """Функция ручной разметки боксов с поддержкой Drag-and-Drop и Backspace"""
    image_original = image.copy()
    image_working = image.copy()
    
    template_coords = []  # Список для хранения [x1, y1, x2, y2]
    drawing = False        # Флаг процесса рисования
    ix, iy = -1, -1       # Стартовые координаты мыши

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, template_coords, image_working
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = image_original.copy()
                # Рисуем все ранее сохраненные прямоугольники
                for rect in template_coords:
                    cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                # Рисуем рамку, которую тянем прямо сейчас (синим цветом для отличия)
                cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
                cv2.imshow("Selecting templates", img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Защита от обратного рисования (справа-налево или снизу-вверх)
            x1, x2 = min(ix, x), max(ix, x)
            y1, y2 = min(iy, y), max(iy, y)
            
            # Исключаем случайные клики без изменения размера
            if (x2 - x1) > 3 and (y2 - y1) > 3:
                template_coords.append([x1, y1, x2, y2])
                print(f"✓ Добавлен прямоугольник {len(template_coords)}: [{x1}, {y1}, {x2}, {y2}]")
            update_image()

    def update_image():
        nonlocal image_working
        image_working = image_original.copy()
        for rect in template_coords:
            cv2.rectangle(image_working, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        cv2.imshow("Selecting templates", image_working)

    # Основной блок выполнения разметки
    cv2.imshow("Selecting templates", image_working)
    cv2.setMouseCallback("Selecting templates", draw_rectangle)
    
    print("\nИнструкция по разметке:")
    print("  - Зажмите ЛКМ и тяните мышку, чтобы выделить объект.")
    print("  - Нажмите BACKSPACE, чтобы удалить последнюю рамку.")
    print("  - Нажмите ENTER, чтобы завершить выбор и запустить подсчет.\n")

    if num_box != None:
        print(f'Выделите ровно {num_box} боксов!!!')
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        elif key == 8:  # Backspace
            if template_coords:
                removed = template_coords.pop()
                print(f"✗ Удален прямоугольник {len(template_coords) + 1}: {removed}")
                print(f"  Осталось прямоугольников: {len(template_coords)}")
                update_image()
            else:
                print("Нет прямоугольников для удаления")
                
    cv2.destroyAllWindows()

    n = len(template_coords)

    if num_box != None and n != num_box:
        raise ValueError("Неккоректное число боксов!")

    print(f"\nИтого выделено примеров: {n}")
    return template_coords

def FamNetTest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    if 'FamNet.test' not in sys.modules:
        from FamNet.test import FamNetMain

    FamNetMain(image, get_box)

def LocaTest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    if 'loca.predict' not in sys.modules:
        from loca.predict import LocaMain

    LocaMain(image, get_box)

def SamClipTest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    print(123)

def SAFECountTest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    if 'SAFECount.tools.test' not in sys.modules:
        from SAFECount.tools.test import SAFECountMain

    SAFECountMain(image, get_box)

def FamNettest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass

def FamNettest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass


if __name__ == "__main__":

    print("Введите путь к изображению или 0 чтобы использовать пример")
    image_path = input()
    if image_path == "0":
        print("Выбран пример")
        image_path = "assets/5.jpg"

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Не удалось считать изображение")

    models = {'1': FamNetTest, '2':LocaTest, '6': SAFECountTest}
    print("Доступные pipeline:")
    for k, v in models.items():
        print(f'Введите {k} для запуска {v.__name__}')
    
    print('\nВведите номер модели:\n')

    i = input()
    mdl = models.get(i, None)

    if mdl is None:
        raise ValueError("Некоректный ввод")

    mdl(image)
