import sys
import os
import argparse
import cv2
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from FamNet.model import CountRegressor, Resnet50FPN
from FamNet.utils import MAPS, Scales, Transform, extract_features
from FamNet.utils import MincountLoss, PerturbationLoss

from Segmentation.OtsuSegmentation import otsu
from Segmentation.GrabCutSegmentation import grabcut
from Segmentation.NoiseSubpression import no_noise
from Segmentation.SAM.SAM import sam_test

def FamNetMain(image, get_box, model_path="FamNet/data/pretrainedModels/FamNet_Save.pth", sam_seg=False, otsu_seg=False, grabcut_seg=False, adapt=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    h_orig, w_orig = image.shape[:2]

    # Инициализация моделей FamNet
    print("Загрузка моделей...")
    resnet50_conv = Resnet50FPN().to(device)
    regressor = CountRegressor(6, pool='mean').to(device)
    
    regressor.load_state_dict(torch.load(model_path, map_location=device))
    resnet50_conv.eval()
    regressor.eval()

    rects_xy = []  # Для отрисовки [x1, y1, x2, y2]
    rects_yx = []  # Для FamNet [y1, x1, y2, x2]

    rects_xy = get_box(image)
    
    # Конвертация и сохранение в формате FamNet (y1, x1, y2, x2)
    for (x1, y1, x2, y2) in rects_xy:
        rects_yx.append([y1, x1, y2, x2])

    if not rects_yx:
        print("Нет рамок для подсчета. Выход.")
        return

    if sam_seg:
        image = sam_test(image, abs(rects_yx[0][2] - rects_yx[0][0]), abs(rects_yx[0][3] - rects_yx[0][1]))
    if grabcut_seg:
        image = grabcut(image)
    if otsu_seg:
        image = otsu(image, 2).astype(np.uint8)

    image_pil = Image.fromarray(image)
    sample = {'image': image_pil, 'lines_boxes': rects_yx}
    sample = Transform(sample)
    image_pil, boxes = sample['image'], sample['boxes']

    if device.type == "cuda":
        image_pil = image.to(device)
        boxes = boxes.to(device)

    # Извлечение признаков (Feature Extraction)
    print("Извлечение признаков (ResNet50)...")
    with torch.no_grad():
        features = extract_features(resnet50_conv, image_pil.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

    # Инференс и Адаптация (Test-time adaptation)
    if not adapt:
        print("Выполнение подсчета (без адаптации)...")
        with torch.no_grad(): 
            output = regressor(features)
    else:
        use_gpu = True if device.type == "cuda" else False
        gradient_steps = 100
        learning_rate = 1e-7
        weight_mincount = 1e-9
        weight_perturbation = 1e-4
        print(f"Запуск адаптации ({gradient_steps} шагов)...")
        features.requires_grad = True
        adapted_regressor = regressor
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=learning_rate)

        pbar = tqdm(range(gradient_steps))
        for step in pbar:
            optimizer.zero_grad()
            output = adapted_regressor(features)
            lCount = weight_mincount * MincountLoss(output, boxes, use_gpu=use_gpu)
            lPerturbation = weight_perturbation * PerturbationLoss(output, boxes, sigma=8, use_gpu=use_gpu)
            Loss = lCount + lPerturbation
            
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()

            pbar.set_description('Adaptation step: {:<3}, loss: {:.4f}, pred-count: {:6.1f}'.format(
                step, Loss.item(), output.sum().item()))

        features.requires_grad = False
        output = adapted_regressor(features)

    # Итоговый результат
    pred_cnt = output.sum().item()
    print(f'\n=================================')
    print(f'ПРЕДСКАЗАННОЕ КОЛИЧЕСТВО: {pred_cnt:.2f}')
    print(f'=================================\n')

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Извлекаем карту плотности в виде 2D numpy массива
    density_map = output[0, 0].detach().cpu().numpy()

    # Создаем общее окно
    plt.figure(figsize=(12, 6))

    # Левая половина: исходная картинка + рамки
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    for box in rects_xy:
        # box: [x1, y1, x2, y2]. Для Rectangle нужны: (x1, y1), ширина, высота
        plt.gca().add_patch(plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            edgecolor='red',
            facecolor='none',
            lw=2
        ))
    plt.title("Исходное изображение (Примеры)")
    plt.axis('off')

    # Правая половина: Карта плотности с результатом в заголовке
    plt.subplot(1, 2, 2)
    # extent=[0, w_orig, h_orig, 0] автоматически растянет/сожмет карту плотности под размеры оригинала
    plt.imshow(density_map, cmap='jet', extent=[0, w_orig, h_orig, 0])
    plt.title(f"Результат: {pred_cnt:.2f}")
    
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    print("Окно визуализации открыто. Закройте окно Matplotlib, чтобы завершить программу.")
    plt.show()

