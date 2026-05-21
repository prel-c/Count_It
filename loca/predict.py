import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from models.loca import build_model

IMG_PATH = "FamNet/data/images_384_VarV2/5.jpg"
MODEL_PATH = "Loca/models/loca_few_shot.pt"
ZERO_SHOT = False 

def run_prediction():
    device = torch.device('cpu')
    
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
        zero_shot = ZERO_SHOT
        backbone_lr = 0.0
        dropout = 0.1
        tiling_p = 0.0

    model = build_model(Args).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        print(f"Ошибка: не удалось загрузить фото {IMG_PATH}")
        return

    my_bboxes = []
    for i in range(3):
        roi = cv2.selectROI(f"Select Object {i+1}", img_bgr, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi
        # Преобразуем формат OpenCV [x, y, w, h] в формат LOCA [x1, y1, x2, y2]
        my_bboxes.append([x, y, x + w, y + h])
        cv2.destroyWindow(f"Select Object {i+1}")

    h_orig, w_orig, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
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
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    for box in my_bboxes:
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                          edgecolor='red', facecolor='none', lw=2))
    plt.title("")

    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(density_map, cmap='jet', extent=[0, w, h, 0])
    plt.title(f"Результат {count:.2f})")

    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_prediction()
