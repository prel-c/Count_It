import cv2
from test import test
from colorama import Fore, Style
from tqdm import tqdm
import numpy as np
hhhh=[]
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

# Сначала определим, сколько всего файлов существует
total_valid = 0
for i in range(8000):
    try:
        with open(f"/mnt/c/CV/FSC147_modifications/FSC147F70/annotations/{i}.txt", 'r') as file:
            total_valid += 1
    except:
        pass

print(f"Найдено {total_valid} аннотаций")
print("-" * 80)

# Создаём прогресс-бар
pbar = tqdm(range(8000), desc="Обработка", unit="img")

for i in pbar:
    try:
        with open(f"/mnt/c/CV/FSC147_modifications/FSC147F70/annotations/{i}.txt", 'r') as file:
            data = file.readlines()
        
        coords = [int(x) for x in data[0].split(",")]
        real_count = int(data[1])
        
        image_BGR = cv2.imread(f"/mnt/c/CV/FSC147_modifications/FSC147F70/images_384_VarV2/{i}.jpg")
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
        if error / real_count>0.7:
            hhhh.append(i)
        
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
        
    except FileNotFoundError:
        continue
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

print(hhhh)