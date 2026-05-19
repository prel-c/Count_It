import os
from PIL import Image
import cv2
import json
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
from test import test

#если у вас не установленны какие-то библиотеки, то запустите
#pip install opencv-python torch tqdm pandas numpy matplotlib colorama pillow


def test1(device, data_path, names_images, row_colors):
    'Основаная тестирующая функция'

    #пути внутри датасета
    anno_file = data_path + 'annotation_FSC147_384.json'
    data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
    im_dir = data_path + 'images_384_VarV2'

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    count = 0 #счётчик картинкок
    SAE = 0 #линейная сумма ошибки
    SSE = 0 #квадратичная сумма ошибки
    SAPE = 0 #сумма относительной ошибки

    print(f"Тестирование на выборке: {names_images}, используется {device}")
    im_ids = data_split[names_images]
    pbar = tqdm(im_ids)
    #проход по всем картинкам
    for im_id in pbar:
        anno = annotations[im_id]
        boxes = anno['box_examples_coordinates']
        dots = np.array(anno['points']) #точки, которыми отмечен каждый объект
        number_object = dots.shape[0] #число объектов

        examples = list()
        for box in boxes:
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[2][0], box[2][1]
            examples.append([y1, x1, y2, x2])

        #image = Image.open(f"{im_dir}, {im_id}") #если нужно считать изображене через PIL

        image = cv2.imread(os.path.join(im_dir, im_id)) #если нужно считать изображене через openCV
        image_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if device.type == "cuda":
            image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).to(device)
            examples = torch.tensor(examples, dtype=torch.float32, device=device)
        y1, x1, y2, x2=examples[0]
        template = image_gr[y1:y2, x1:x2]
        image2=image.copy()
        #cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.imwrite(f"/mnt/c/CV/Test/results/{im_id}.jpg", image2)
        output, image1 = test(image_gr, template, image)
        cv2.imwrite(f"/mnt/c/CV/Test/results/{im_id}.jpg", image1)

        #подсчёт ошибок
        pred_number_object = output
        count += 1
        error = abs(number_object - pred_number_object)
        SAE += error
        SSE += error**2
        if number_object > 0:
            SAPE += (error)/(number_object)

        MAE = SAE/count
        RMSE = (SSE/count) ** 0.5
        MAPE = SAPE/count

        #раскрашиваение ошибок и строк (пороги определяют когда красить в красный и жёлтый)
        base_color = row_colors[count % 2]
        er_color = error_color(error, porog1=50, porog2=25)
        mae_color = error_color(MAE, porog1=35, porog2=25)
        rmse_color = error_color(RMSE, porog1=125, porog2=100)
        mape_color = error_color(MAPE, porog1=1, porog2=0.45)

        #красивый вывод строк
        pbar.set_description(f'{base_color}{im_id:<10}'
                            f' number obj: {number_object:4d}, pred: {pred_number_object:4d}, '
                            f'{er_color}error: {error:4d}{base_color}. '
                            f'{mae_color}MAE: {MAE:6.2f}{base_color}, '
                            f'{rmse_color}RMSE: {RMSE:6.2f}{base_color}, '
                            f'{mape_color}MAPE: {MAPE:6.2%}{base_color}')
        pbar.refresh()

    print(f'On {names_images} data, MAE: {mae_color}{MAE:6.2f}{base_color}, RMSE: {rmse_color}{RMSE:6.2f}{base_color}, MAPE: {mape_color}{MAPE:6.2%}{base_color}')

def primer_function_pre_processing(im, ex):
    "пред-обработка"

    image = im

    return image

def primer_function_for_counting(im, ex, num):
    "считает объекты и рисует примеры на изображении"

    count = abs(num + np.random.randint(-75, 75))


    #если раскоментить, то будет рисовать боксы примеров на изображении
    """
    image = im.copy()

    if len(ex) >= 1:
        for y1, x1, y2, x2 in ex:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Найдено объектов: {count}")
    plt.show()
    """

    return count

def primer_function_post_processing(im, ex, cnt):
    "пост-обработка - если кол-во кратно 2, то вычитается 2"
    c = cnt
    if cnt % 2 == 0:
        c = cnt - 2
    return c

def error_color(error, porog1=0, porog2=0):
    "раскрашивает ошибки"

    if error > porog1:
        color = Fore.RED
    elif error > porog2:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN

    return color


if __name__ == "__main__":
    #"cuda" if torch.cuda.is_available() else 
    device = torch.device("cpu")

    data_path = '/mnt/c/CV/FSC147_modifications/FSC147O/'

    names_images = 'test' #выбор по какой выборке из датасета делать тестирование
    #выборки (список имён изображений) записанны в файле Train_Test_Val_FSC_147.json
    #можно написать там свою выборку и указать её название в names_images
    #например можно сделать выборку из фото с 300 по 700 и записать их под ключём
    #"my" в Train_Test_Val_FSC_147.json, тогда в names_images указать "my"

    row_colors = [Fore.WHITE, Fore.LIGHTBLACK_EX] #цвета для раскрашивая строк, белый и серый

    test1(device, data_path, names_images, row_colors)