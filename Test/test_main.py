import cv2
from test_classic import test
from colorama import Fore, Style
from tqdm import tqdm
import numpy as np
import os

def error_color(error, porog1=50, porog2=25):
    "раскрашивает ошибки"
    if error > porog1:
        color = Fore.RED
    elif error > porog2:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN
    return color

# Инициализация
sae = 0
sse = 0
sape = 0
count = 0

print("Тестирование на тестовой выборке")
print("-" * 80)

row_colors = [Fore.WHITE, Fore.LIGHTBLACK_EX]

# Получаем список существующих аннотаций
annotation_dir = "/mnt/c/CV/FSC147_modifications/FSC147/annotations/"
image_dir = "/mnt/c/CV/FSC147_modifications/FSC147/images_384_VarV2/"

# Собираем существующие индексы (только те, для которых есть и аннотация, и изображение)
valid_indices = []
for filename in os.listdir(annotation_dir):
    if filename.endswith('.txt'):
        idx = int(filename.split('.')[0])
        # Проверяем, существует ли соответствующее изображение
        if os.path.exists(f"{image_dir}/{idx}.jpg"):
            valid_indices.append(idx)

valid_indices.sort()  # Сортируем для порядка
total_valid = len(valid_indices)

print(f"Найдено {total_valid} изображений")
print("-" * 80)

# Создаём прогресс-бар с правильным количеством
pbar = tqdm(valid_indices, desc="Обработка", unit="img")

for i in pbar:
    try:
        with open(f"{annotation_dir}/{i}.txt", 'r') as file:
            data = file.readlines()
        
        coords = [int(x) for x in data[0].split(",")]
        real_count = int(data[1])
        
        image_BGR = cv2.imread(f"{image_dir}/{i}.jpg")
        if image_BGR is None:
            continue
            
        image1 = image_BGR.copy()
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
        template = image[coords[1]:coords[3], coords[0]:coords[2]]
        
        pred_count, image2 = test(image, template, image1)
        cv2.imwrite(f"results/{i}.jpg", image2)
        
        # Подсчёт ошибок
        count += 1
        error = abs(pred_count - real_count)
        sae += error
        sse += error ** 2
        if real_count > 0:
            sape += error / real_count
        
        # Текущие метрики
        MAE = sae / count
        RMSE = (sse / count) ** 0.5
        MAPE = (sape / count) * 100 if count > 0 else 0
        
        # Раскрашивание
        base_color = row_colors[count % 2]
        er_color = error_color(error, porog1=50, porog2=25)
        mae_color = error_color(MAE, porog1=35, porog2=25)
        rmse_color = error_color(RMSE, porog1=125, porog2=100)
        mape_color = error_color(MAPE, porog1=100, porog2=45)
        
        # Обновление описания прогресс-бара
        pbar.set_description(
            f'{base_color}IMG_{i:4d} | real: {real_count:3d} pred: {pred_count:3d} | '
            f'{er_color}err: {error:3d}{base_color} | '
            f'{mae_color}MAE: {MAE:5.2f}{base_color} | '
            f'{mape_color}MAPE: {MAPE:5.2f}%{base_color}'
        )
        pbar.refresh()
        
    except Exception as e:
        pbar.set_description(f"{Fore.YELLOW}Ошибка {i}: {e}{Style.RESET_ALL}")
        continue

# Финальные результаты
print("-" * 80)
if count > 0:
    MAE = sae / count
    RMSE = (sse / count) ** 0.5
    MAPE = (sape / count) * 100
    
    print(f'\n{Fore.CYAN}{"="*50}{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ{Style.RESET_ALL}')
    print(f'{Fore.CYAN}{"="*50}{Style.RESET_ALL}')
    print(f'Обработано изображений: {count}')
    print(f'{Fore.GREEN}MAE:  {MAE:.2f}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}RMSE: {RMSE:.2f}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}MAPE: {MAPE:.2f}%{Style.RESET_ALL}')
    print(f'{Fore.CYAN}{"="*50}{Style.RESET_ALL}')
else:
    print(f'{Fore.RED}НЕ ОБРАБОТАНО НИ ОДНОГО ИЗОБРАЖЕНИЯ!{Style.RESET_ALL}')