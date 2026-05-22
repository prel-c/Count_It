import sys
import os
import argparse
import cv2
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_helper import build_network

def clean_state_dict(state_dict):
    """Очищает ключи весов от приставки 'module.', если модель обучалась через DDP"""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

def SAFECountMain(image, get_box, config="SAFECount/experiments/FSC147/config.yaml", weights="SAFECount/tools/ckpt_best.pth"):

    with open(config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    print("Загрузка модели...")
    model = build_network(config.net)
    
    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    
    model.to(device)
    model.eval()
   
    boxes = get_box(image)
    
    if len(boxes) == 0:
        print("Вы не выделили ни одного объекта. Выход.")
        return

    h_orig, w_orig = image.shape[:2]
    h_new = int(np.ceil(h_orig / 32.0) * 32)
    w_new = int(np.ceil(w_orig / 32.0) * 32)
    
    print(f"Изменяем размер изображения с {w_orig}x{h_orig} на {w_new}x{h_new} (кратно 32)")
    
    img_resized = cv2.resize(image, (w_new, h_new))
    
    boxes_resized = []
    for (x1, y1, x2, y2) in boxes:
        x1_n = x1 * (w_new / w_orig)
        x2_n = x2 * (w_new / w_orig)
        y1_n = y1 * (h_new / h_orig)
        y2_n = y2 * (h_new / h_orig)
        boxes_resized.append([y1_n, x1_n, y2_n, x2_n])

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0).to(device) # Shape: [1, 3, H_new, W_new]
    boxes_tensor = torch.tensor(boxes_resized, dtype=torch.float32).unsqueeze(0).to(device) # Shape: [1, N, 4]

    sample = {
        'image': img_tensor,
        'boxes': boxes_tensor
    }

    sample = {
        'image': img_tensor,
        'boxes': boxes_tensor
    }

    print("Выполнение подсчета...")
    with torch.no_grad():
        outputs = model(sample)
        
        density_pred = outputs["density_pred"]
        pred_cnt = torch.sum(density_pred).item()

    print(f"\n=================================")
    print(f"ПРЕДСКАЗАННОЕ КОЛИЧЕСТВО: {pred_cnt:.2f}")
    print(f"=================================\n")
    

    density_map = density_pred[0, 0].detach().cpu().numpy()


    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    plt.imshow(image)
    for box in boxes:
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

    plt.subplot(1, 2, 2)

    plt.imshow(density_map, cmap='jet', extent=[0, w_orig, h_orig, 0])
    plt.title(f"Результат: {pred_cnt:.2f}")
    
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    print("Окно визуализации открыто. Закройте окно Matplotlib, чтобы завершить программу.")
    plt.show()
