import cv2
import numpy as np

coords = []                                                                             # Координаты бокса
drawing = False                                                                         # По умолчанию не рисуем
ix, iy = -1, -1                                                                         # Начальное положение курсора
template_coords = []                                                                    # Список для координат боксов шаблонов
image_original = None                                                                   # Сохраняем оригинал изображения

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, coords, template_coords, image
    
    if event == cv2.EVENT_LBUTTONDOWN:                                                  # Если нажата левая кнопка мыши, то рисуем
        drawing = True
        ix, iy = x, y
        coords = [(ix, iy)]
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = image_original.copy()                                            
            for rect in template_coords:                                                # Рисуем все сохраненные прямоугольники
                cv2.rectangle(img_copy, rect[0], rect[1], (0, 255, 0), 2)
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)                   # Рисуем текущий прямоугольник
            cv2.imshow("Selecting templates", img_copy)
            
    elif event == cv2.EVENT_LBUTTONUP:                                                  # Если не нажата левая кнопка мыши, то не рисуем
        drawing = False
        coords.append((x, y))                                                           # Запоминаем последние координаты курсора
        template_coords.append([coords[0], coords[1]])                                  # Сохраняем прямоугольник
        print(f"✓ Добавлен прямоугольник {len(template_coords)}: {coords}")            
        update_image()                                                                  # Обновляем изображение

def update_image():                                                                     # Функция для обновления всех прямоугольников
    global image
    image = image_original.copy()
    for rect in template_coords:
        cv2.rectangle(image, rect[0], rect[1], (0, 255, 0), 2)
    cv2.imshow("Selecting templates", image)

image_original = cv2.imread("image.jpg")                                                # Загружаем изображение
image = image_original.copy()

if image_original is None:                                                              # Прверка, есть ли изображение в папке
    print("Ошибка: не удалось загрузить изображение")
else:
    cv2.imshow("Selecting templates", image)                                            # Выводим картинку для выбора шаблона
    cv2.setMouseCallback("Selecting templates", draw_rectangle)                         # Выбираем шаблоны
    
    print("Инструкция:")
    print("- Нажмите и перетащите мышку для выделения прямоугольника")
    print("- Нажмите BACKSPACE для удаления последнего прямоугольника")
    print("- Нажмите Enter для завершения")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:                                                                   # Enter для выхода
            break
        elif key == 8:                                                                  # BACKSPACE для удаления последнего прямоугольника
            if template_coords:
                removed = template_coords.pop()                                         # Удаляем последний прямоугольник из общего списка
                print(f"✗ Удален последний прямоугольник {len(template_coords) + 1}: {removed}")
                print(f"  Осталось прямоугольников: {len(template_coords)}")
                update_image()
            else:
                print("Нет прямоугольников для удаления")
    
    cv2.destroyAllWindows()

print(f"\nИтоговые координаты ({len(template_coords)} прямоугольников):")
for i, rect in enumerate(template_coords):
    print(f"{i+1}. {rect}")