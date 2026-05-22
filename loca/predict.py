import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from loca.models.loca import build_model


def LocaMain(image, get_box, model_path="Loca/models/loca_few_shot.pt", zero=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class Args:
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
        zero_shot = zero
        backbone_lr = 0.0
        dropout = 0.1
        tiling_p = 0.0

    model = build_model(Args).to(device)
    state_dict = torch.load(model_path, map_location=device)['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    my_bboxes = get_box(image, 3)

    h_orig, w_orig, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    bboxes = torch.tensor(my_bboxes).float()
    bboxes[:, [0, 2]] *= (512.0 / w_orig)
    bboxes[:, [1, 3]] *= (512.0 / h_orig)
    bboxes = bboxes.unsqueeze(0).to(device)

    with torch.no_grad():
        output, _ = model(img_tensor, bboxes)
    
    count = output.sum().item()
    print(f"Итоговое количество: {count:.2f}")

    h, w = img_rgb.shape[:2]
    density_map = output[0].squeeze().cpu().numpy()


    # Визуализация в одном окне через Matplotlib
    # Переводим исходную картинку в RGB для корректных цветов в plt
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Создаем общее окно
    plt.figure(figsize=(12, 6))

    # Левая половина: исходная картинка + рамки
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    for box in my_bboxes:
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
    plt.title(f"Результат: {count:.2f}")
    
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    print("Окно визуализации открыто. Закройте окно Matplotlib, чтобы завершить программу.")
    plt.show()
