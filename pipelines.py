import warnings
warnings.filterwarnings("ignore")
import os
import gc
import copy
from PIL import Image
import cv2
import json
import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

#from test.main import draw_rectangle, update_image

from FamNet.model import CountRegressor, Resnet50FPN
from FamNet.utils import MAPS, Scales, Transform, extract_features
from FamNet.utils import MincountLoss, PerturbationLoss, visualize_output_and_save, select_exemplar_rois

from loca.models.loca import build_model

from Segmentation.OtsuSegmentation import otsu
from Segmentation.GrabCutSegmentation import grabcut
from Segmentation.NoiseSubpression import no_noise
from Segmentation.SAM.SAM import sam_test

from YOLO.yolopipe import *

image_original = cv2.imread("5.jpg")                                                # Загружаем изображение
image = image_original.copy()

def box(image):
    coords = []                                                                             # Координаты бокса
    drawing = False                                                                         # По умолчанию не рисуем
    ix, iy = -1, -1                                                                         # Начальное положение курсора
    template_coords = []                                                                    # Список для координат боксов шаблонов
    image_original = None                                                                   # Сохраняем оригинал изображения

    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing, coords, template_coords, image
        
        if event == cv2.EVENT_LBUTTONDOWN:                                                  # Если нажата левая кнопка мыши, то рисуем
            drawing = True
            ix, iy = x, y
            coords = [(ix, iy)]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = image_original.copy()                                            
                for rect in template_coords:                                                # Рисуем все сохраненные прямоугольники
                    cv2.rectangle(img_copy, rect[0], rect[1], (0, 255, 0), 2)
                cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)                   # Рисуем текущий прямоугольник
                cv2.imshow("Selecting templates", img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:                                                  # Если не нажата левая кнопка мыши, то не рисуем
            drawing = False
            coords.append((x, y))                                                           # Запоминаем последние координаты курсора
            template_coords.append([coords[0], coords[1]])                                  # Сохраняем прямоугольник
            print(f"✓ Добавлен прямоугольник {len(template_coords)}: {coords}")            
            update_image()                                                                  # Обновляем изображение

    def update_image():                                                                     # Функция для обновления всех прямоугольников
        global image
        image = image_original.copy()
        for rect in template_coords:
            cv2.rectangle(image, rect[0], rect[1], (0, 255, 0), 2)
        cv2.imshow("Selecting templates", image)

        if image_original is None:                                                              # Прверка, есть ли изображение в папке
            print("Ошибка: не удалось загрузить изображение")
        else:
            cv2.imshow("Selecting templates", image)                                            # Выводим картинку для выбора шаблона
            cv2.setMouseCallback("Selecting templates", draw_rectangle)                         # Выбираем шаблоны
            
            print("Инструкция:")
            print("- Нажмите и перетащите мышку для выделения прямоугольника")
            print("- Нажмите BACKSPACE для удаления последнего прямоугольника")
            print("- Нажмите Enter для завершения")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:                                                                   # Enter для выхода
                    break
                elif key == 8:                                                                  # BACKSPACE для удаления последнего прямоугольника
                    if template_coords:
                        removed = template_coords.pop()                                         # Удаляем последний прямоугольник из общего списка
                        print(f"✗ Удален последний прямоугольник {len(template_coords) + 1}: {removed}")
                        print(f"  Осталось прямоугольников: {len(template_coords)}")
                        update_image()
                    else:
                        print("Нет прямоугольников для удаления")
            
            cv2.destroyAllWindows()

            print(f"\nИтоговые координаты ({len(template_coords)} прямоугольников):")
            for i, rect in enumerate(template_coords):
                print(f"{i+1}. {rect}")
    return template_coords

def FamNetTest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    model_path = "./FamNet/data/pretrainedModels/FamNet_Save.pth"

    resnet50_conv = Resnet50FPN()
    regressor = CountRegressor(6, pool='mean')

    regressor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    resnet50_conv.to(device)
    regressor.to(device)

    resnet50_conv.eval()
    regressor.eval()

    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]


    out_bbox_file = "{}/{}_box.txt".format('.', image_name)
    fout = open(out_bbox_file, "w")
    im = cv2.imread(image_path, cv2.IMREAD_COLOR)  # игнорирует альфа-канал
    #im = cv2.imread(args.input_image)
    cv2.imshow('image', im)
    rects = select_exemplar_rois(im)

    rects1 = list()

    print("Bounding boxes: ", end="")
    print(rects1)

    image = Image.open(image_path).convert('RGB')
    image.load()
    sample = {'image': image, 'lines_boxes': rects1}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']

    with torch.no_grad():
        features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
        output = regressor(features)

    print('===> The predicted count is: {:6.2f}'.format(output.sum().item()))

    np.save('density_map_result.npy', output.cpu().detach().numpy())

    rslt_file = "{}/{}_out.png".format(".", image_name)
    visualize_output_and_save(image.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file)
    print("===> Visualized output is saved to {}".format(rslt_file))


def LocaTest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    loca_model_path = "./loca/models/loca_few_shot.pt"
    loca_model = build_model(LocaArgs).to(device)
    state_dict = torch.load(loca_model_path, map_location=device)['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    loca_model.load_state_dict(state_dict)
    loca_model.eval()

    # Трансформации строго для Loca
    loca_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def SamClipTest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass

def Test(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass

def FamNettest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass

def FamNettest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass

def FamNettest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass

def FamNettest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass

def FamNettest(adapt=False, otsu_seg=False, grabcut_seg=False, sam_seg=False, th=False, image_path=None, box=None):
    pass

FamNetTest(image_path = "5.jpg")