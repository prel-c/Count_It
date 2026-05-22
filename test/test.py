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

from FamNet.model import CountRegressor, Resnet50FPN
from FamNet.utils import MAPS, Scales, Transform, extract_features
from FamNet.utils import MincountLoss, PerturbationLoss

from loca.models.loca import build_model

from Segmentation.OtsuSegmentation import otsu
from Segmentation.GrabCutSegmentation import grabcut
from Segmentation.NoiseSubpression import no_noise
from Segmentation.SAM.SAM import sam_test

class LocaArgs:
    backbone = 'resnet50'
    swav_backbone = True
    reduction = 8
    image_size = 512
    num_enc_layers = 3
    num_ope_iterative_steps = 3
    emb_dim = 256
    num_heads = 8
    kernel_dim = 3
    num_objects = 3
    pre_norm = True
    zero_shot = False
    backbone_lr = 0.0
    dropout = 0.1
    tiling_p = 0.0


def test(device, data_path, names_images, row_colors, primer, adapt, otsu_seg, grabcut_seg, sam_seg, th, model_type=None):
    'Основаная тестирующая функция'

    #пути внутри датасета
    anno_file = data_path + 'annotation_FSC147_384.json'
    data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
    im_dir = data_path + 'images_384_VarV2'

    if model_type == "FamNet":
        model_path = "./FamNet/data/pretrainedModels/FamNet_Save.pth"

        resnet50_conv = Resnet50FPN()
        resnet50_conv.to(device)
        resnet50_conv.eval()

        regressor = CountRegressor(6, pool='mean')
        regressor.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        regressor.to(device)
        regressor.eval()

        gradient_steps = 100
        learning_rate = 1e-7
        weight_mincount = 1e-9
        weight_perturbation = 1e-4

        adapted_regressor = copy.deepcopy(regressor).to(device)
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=learning_rate)
        base_state = copy.deepcopy(adapted_regressor.state_dict())

    elif model_type == "Loca":
        if adapt == True:
            print("Пропуск конфигурации с адаптацией для Loca")
            return

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
    else:
        raise ValueError("Неизвестный тип модели. Выберите 'famnet' или 'loca'.")

    with open(anno_file) as f:
        annotations = json.load(f)

    print(123131231, len(annotations.keys()))

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
            if model_type == "FamNet":
                examples.append([y1, x1, y2, x2])
            else:
                examples.append([x1, y1, x2, y2])

        #image = Image.open(os.path.join(im_dir, im_id)) #если нужно считать изображене через PIL

        image = cv2.imread(os.path.join(im_dir, im_id))#если нужно считать изображене через openCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig, _ = image.shape
        
        if primer:
            output = primer_function_for_counting(image, examples, 1)
        else:
            if sam_seg:
                image = sam_test(image, abs(examples[0][2] - examples[0][0]), abs(examples[0][3] - examples[0][1]))
            if grabcut_seg:
                image = grabcut(image)
            if otsu_seg:
                image = otsu(image, 2).astype(np.uint8)


            if model_type == "FamNet":

                image = Image.fromarray(image)

                sample = {'image': image, 'lines_boxes': examples}
                sample = Transform(sample)
                image, boxes = sample['image'], sample['boxes']
                
                if device.type == "cuda":
                    image = image.to(device)
                    boxes = boxes.to(device)

                with torch.no_grad():
                    features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

                if not adapt:
                    with torch.no_grad():
                        output = regressor(features)

                else:
                    features = features.detach().to(device)
                    adapted_regressor.load_state_dict(base_state)
                    optimizer = torch.optim.Adam(adapted_regressor.parameters(), lr=learning_rate)
                    
                    for step in range(0, gradient_steps):
                        optimizer.zero_grad()
                        output = adapted_regressor(features)
                        lCount = weight_mincount * MincountLoss(output, boxes)
                        lPerturbation = weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
                        Loss = lCount + lPerturbation
                        Loss.backward()
                        optimizer.step()
                    with torch.no_grad():
                        output = adapted_regressor(features)
            
            else:
                # Трансформации для Loca
                img_tensor = loca_transform(image).unsqueeze(0).to(device)

                # Масштабирование рамок под 512x512
                bboxes = torch.tensor(examples).float()

                # Корректируем количество рамок под требования архитектуры Loca
                if bboxes.shape[0] > LocaArgs.num_objects:
                    # Если рамок больше (например, 4), берем строго первые 3
                    bboxes = bboxes[:LocaArgs.num_objects]

                elif bboxes.shape[0] < LocaArgs.num_objects:
                    # Если вдруг рамок меньше (например, 1 или 2), дублируем последнюю рамку до нужного количества
                    num_missing = LocaArgs.num_objects - bboxes.shape[0]        
                    last_box = bboxes[-1:].repeat(num_missing, 1)
                    bboxes = torch.cat([bboxes, last_box], dim=0)

                bboxes[:, [0, 2]] *= (512.0 / w_orig)
                bboxes[:, [1, 3]] *= (512.0 / h_orig)
                bboxes = bboxes.unsqueeze(0).to(device)

                with torch.no_grad():
                    output, _ = loca_model(img_tensor, bboxes)
            
            if th:
                output = no_noise(output).sum()
                print(123)
            else:
                output = output.sum()
            

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
                            f' number obj: {number_object:4d}, pred: {pred_number_object:6.2f}, '
                            f'{er_color}error: {error:6.2f}{base_color}. '
                            f'{mae_color}MAE: {MAE:6.2f}{base_color}, '
                            f'{rmse_color}RMSE: {RMSE:6.2f}{base_color}, '
                            f'{mape_color}MAPE: {MAPE:6.2%}{base_color}')

        torch.cuda.empty_cache()

    print(f'On {names_images} data, MAE: {mae_color}{MAE:6.2f}{base_color}, RMSE: {rmse_color}{RMSE:6.2f}{base_color}, MAPE: {mape_color}{MAPE:6.2%}{base_color}')
    
def error_color(error, porog1=0, porog2=0):
    "раскрашивает ошибки"

    if error > porog1:
        color = Fore.RED
    elif error > porog2:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN

    return color

def primer_function_for_counting(im, ex, num):
    "считает объекты и рисует примеры на изображении"

    count = abs(num + np.random.randint(-75, 75))

    return count

if __name__ == "__main__":

    row_colors = [Fore.WHITE, Fore.LIGHTBLACK_EX] #цвета для раскрашивая строк, белый и серый
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = 'FamNet/data/'

    names_images = 'test'

    models_to_test = ['FamNet', 'Loca']
    
    primer = True #включает режим примера (чтобы посмотреть работает ли тестирование)
    adapt = False #включает адаптацию

    #предобработки
    otsu_seg = False
    grabcut_seg = False
    sam_seg = False

    #постобработки
    th = False

    pipelines = [(False, False, False, False, False, False), #проверка
                 (False, False, False, False, False, False), #оригинал
                 (False, False, True, False, False, False),#otsu
                 (False, False, False, True, False, False),#grabcut
                 (False, False, False, False, True, False),#SAM
                 (False, False, False, False, False, True),#th
                 (False, True, False, False, False, False),#adapt
                 (False, True, False, False, False, True),#adapt + th
                 ]
    
    for model in models_to_test:
        print(f"\n{'='*50}\nЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛИ: {model.upper()}\n{'='*50}")
        i = 1
        l = len(pipelines)
        for primer, adapt, otsu_seg, grabcut_seg, sam_seg, th in pipelines:
            print('')
            print(f'Тестирование номер: {i} из {l}. primer = {primer}, adapt = {adapt}, otsu_seg = {otsu_seg}, grabcut_seg = {grabcut_seg}, sam_seg = {sam_seg}, th = {th}')
            test(device, data_path, names_images, row_colors, primer, adapt, otsu_seg, grabcut_seg, sam_seg, th, model_type=model)
            i += 1