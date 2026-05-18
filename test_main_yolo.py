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

#если у вас не установленны какие-то библиотеки, то запустите
#pip install opencv-python torch tqdm pandas numpy matplotlib colorama pillow


def test(device, data_path, names_images, row_colors):
    'Основаная тестирующая функция'

    #пути внутри датасета
    anno_dir = data_path + 'yolo_labels'
    data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
    im_dir = data_path + 'images_384_VarV2'

    with open(data_split_file) as f:
        data_split = json.load(f)

    count = 0 #счётчик картинкок
    SAE = 0 #линейная сумма ошибки
    SSE = 0 #квадратичная сумма ошибки
    SAPE = 0 #сумма относительной ошибки

    print(f"Тестирование на выборке: {names_images}, используется {device}")
    im_ids = data_split[names_images]
    pbar = tqdm(im_ids)
    #проход по всем картинкам и их аннотациям
    for im_id in pbar:

        #image = Image.open(f"{im_dir}, {im_id}") #если нужно считать изображене через PIL

        image = cv2.imread(os.path.join(im_dir, im_id)) #если нужно считать изображене через openCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape

        anno = tuple(tuple(map(float, i.split())) for i in open(os.path.join(anno_dir, im_id[:-4] + ".txt"), 'r').read().strip().split('\n'))
        #print(type(anno), anno)
        number_object = len(anno) #число объектов
        if number_object >= 3:
            boxes = (anno[number_object//2], anno[number_object//3 - 1], anno[number_object//3 * 2 + 1])
            #print(boxes, type(boxes))
        else:
            boxes = anno[0]

        examples = list()
        for box in boxes:
            x_center_norm = box[1]
            y_center_norm = box[2]
            w_norm = box[3]
            h_norm = box[4]

            x_center = x_center_norm * img_w
            y_center = y_center_norm * img_h
            w_pixel = w_norm * img_w
            h_pixel = h_norm * img_h

            x1 = int(x_center - (w_pixel / 2))
            y1 = int(y_center - (h_pixel / 2))
            x2 = int(x_center + (w_pixel / 2))
            y2 = int(y_center + (h_pixel / 2))

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            examples.append([y1, x1, y2, x2])

        if device.type == "cuda":
            image = torch.from_numpy(image).permute(2, 0, 1).to(device)
            examples = torch.tensor(examples, dtype=torch.float32, device=device)

        #вызов функций для обратки и подсчёта
        image_pre_processing = primer_function_pre_processing(image, examples)
        output = primer_function_for_counting(image_pre_processing, examples, number_object)
        output = primer_function_post_processing(image, examples, output)

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
        print('')

    print(f'On {names_images} data, MAE: {mae_color}{MAE:6.2f}{base_color}, RMSE: {rmse_color}{RMSE:6.2f}{base_color}, MAPE: {mape_color}{MAPE:6.2%}{base_color}')

def primer_function_pre_processing(im, ex):
    "пред-обработка"

    image = im

    return image

def primer_function_for_counting(im, ex, num):
    "считает объекты и рисует примеры на изображении"

    count = abs(num + np.random.randint(-75, 75))


    #если раскоментить, то будет рисовать боксы примеров на изображении

    """image = im.copy()

    if len(ex) >= 1:
        for y1, x1, y2, x2 in ex:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Найдено объектов: {count}")
    plt.show()"""


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
    
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))

    data_path = 'FamNet/data/' #путь к датасету

    names_images = 'test' #выбор по какой выборке из датасета делать тестирование
    #выборки (список имён изображений) записанны в файле Train_Test_Val_FSC_147.json
    #можно написать там свою выборку и указать её название в names_images
    #например можно сделать выборку из фото с 300 по 700 и записать их под ключём
    #"my" в Train_Test_Val_FSC_147.json, тогда в names_images указать "my"

    row_colors = [Fore.WHITE, Fore.LIGHTBLACK_EX] #цвета для раскрашивая строк, белый и серый

    test(device, data_path, names_images, row_colors)