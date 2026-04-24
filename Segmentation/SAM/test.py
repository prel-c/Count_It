import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

# 1. Настройка SAM
checkpoint = "Segmentation/SAM/data/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# 2. Загрузка изображения
image_path = "D:/mygit/opd/ResNet/images/image copy 4.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

box = np.array(cv2.selectROI("Select Area", image, fromCenter=False, showCrosshair=True))
cv2.destroyAllWindows()
x, y, w, h = box
input_box = np.array([x, y, x + w, y + h])

# Создаем копию изображения для результата
clean_image = image.copy()

# SAM предсказывает маску конкретно для этого бокса
masks, scores, logits = predictor.predict(
    box=input_box[None, :],
    multimask_output=False
)

# Получаем булеву маску (True - объект, False - фон)
mask = masks[0]

# 4. Очистка фона (создаем черное полотно)
# Если хочешь оставить только объект, а остальное черным:
clean_image = np.zeros_like(image_rgb)
# Копируем только те пиксели, где маска True
clean_image[mask] = image_rgb[mask]

"""x1, y1, x2, y2 = box

# Вырезаем зону бокса из маски и из картинки
box_mask = mask[y1:y2, x1:x2]
box_roi = clean_image[y1:y2, x1:x2]

# Закрашиваем черным (0, 0, 0) все пиксели ВНУТРИ бокса, которые SAM счел фоном
box_roi[~box_mask] = [0, 0, 0] 

# Вставляем очищенный кусок обратно в картинку
clean_image[y1:y2, x1:x2] = box_roi
"""
# 4. Сохраняем или показываем результат
cv2.imwrite("loca/images/676_cleaned.jpg", clean_image)

# Визуализация (чтобы убедиться, что работает)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Оригинал")
plt.imshow(image_rgb)
plt.subplot(1, 2, 2)
plt.title("Очищенные примеры")
plt.imshow(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
plt.show()