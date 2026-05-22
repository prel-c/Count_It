import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

checkpoint = "Segmentation\SAM\data\sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

sam = sam_model_registry[model_type](checkpoint=checkpoint)

print(next(sam.parameters()).device)

# sam.to(device)

image_path = "D:/mygit/opd/ResNet/images/image copy 4.png"
image = cv2.imread(image_path)

if image is None:
    print("Ошибка: не удалось загрузить изображение.")
    exit()

# image = cv2.resize(image, (1024, 1024))

box = np.array(cv2.selectROI("Select Area", image, fromCenter=False, showCrosshair=True))
cv2.destroyAllWindows()
x, y, w, h = box

MIN_AREA = int(0.03 * w * h)    # Отсекаем мелкий шум (блики, пылинки)
MAX_AREA = int(2 * w * h)   # Отсекаем фон (стены, пол, огромные тени)

print(MIN_AREA, MAX_AREA)

#input_box = np.array([x, y, x + w, y + h

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Генерация масок (это может занять несколько секунд)...")
mask_generator = SamAutomaticMaskGenerator(model=sam,
                                           points_per_side=32,
                                           crop_n_layers=1,
                                           points_per_batch=32)
masks = mask_generator.generate(image)
print(f"Найдено масок всего: {len(masks)}")



# Создаем пустое черное полотно для итоговой маски (False)
height, width, _ = image.shape
combined_mask = np.zeros((height, width), dtype=bool)

filtered_count = 0

for mask_data in masks:
    area = mask_data['area']
    print(area)
    
    # Если площадь маски похожа на наш объект - берем её
    if MIN_AREA <= area <= MAX_AREA:
        combined_mask = np.logical_or(combined_mask, mask_data['segmentation'])
        filtered_count += 1

print(f"Осталось масок объектов после фильтрации: {filtered_count}")

# ==========================================
# РАСШИРЕНИЕ МАСКИ (Dilation)
# ==========================================

# 1. Сначала переведем булеву маску в формат uint8 (0 и 255), так как OpenCV работает с ним
mask_uint8 = (combined_mask.astype(np.uint8)) * 255

# 2. Вычисляем размер "отступа". 
# 10% от ширины выбранного ранее бокса (w):
padding = int((w + h) // 2 * 0.1) 
if padding < 1: padding = 1 # Чтобы не было нулевого ядра

# 3. Создаем ядро для расширения
kernel = np.ones((padding, padding), np.uint8)

# 4. Применяем дилатацию
dilated_mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

# 5. Возвращаем маску обратно в булев тип для фильтрации
combined_mask = dilated_mask_uint8 > 0

clean_image = image.copy()

clean_image[~combined_mask] = list(map(int, image.mean(axis=(0, 1))))

#clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("1. Оригинал")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"2. Фильтр масок ({filtered_count} шт.)")
# Показываем саму маску: белое - объекты, черное - фон
plt.imshow(combined_mask, cmap='gray') 
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("3. Очищено для нейросети")
plt.imshow(clean_image)
plt.axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite("D:/mygit/opd/ResNet/images/image copy 222.png", clean_image)
