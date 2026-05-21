import cv2
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Путь к скачанному файлу весов
checkpoint = "Segmentation\SAM\data\sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Загрузка модели
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device)

# Генератор масок
mask_generator = SamAutomaticMaskGenerator(sam)

# Загрузка изображения
image_path = "loca/images/676.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Ошибка: файл test.jpg не найден")
    exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Генерация масок
masks = mask_generator.generate(image)
print(f"Найдено объектов: {len(masks)}")

# Показать изображение с первой маской
plt.imshow(image)
if masks:
    plt.imshow(masks[0]["segmentation"], alpha=0.5)
plt.show()
