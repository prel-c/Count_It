import os
import cv2
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from classic import classic

def get_box(image, num_box=None):                                                                           # Функция для ручной разметки шаблона

    image_original = image
    image_working = image_original.copy()
    
    template_coords = []                                                                                    # Список для хранения [x1, y1, x2, y2]
    drawing = False                                                                                         # Флаг процесса рисования = False
    ix, iy = -1, -1                                                                                         # Стартовые координаты мыши

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, template_coords, image_working                                            # Объявляем все в начале
        
        if event == cv2.EVENT_LBUTTONDOWN:                                                                  # Когда нажата лавая кнопка мыши
            drawing = True                                                                                  # Флаг процесса рисования = True
            ix, iy = x, y                                                                                   # Координаты курсора
            
        elif event == cv2.EVENT_MOUSEMOVE:                                                                  # Если мышь движется
            if drawing:
                img_copy = image_original.copy()                                                            
                for rect in template_coords:                                                                # Рисуем все ранее сохраненные прямоугольники
                    cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)                                   # Рисуем рамку, которую тянем прямо сейчас
                cv2.imshow("Selecting templates", img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:                                                                  # Если левая кнопка мыши отжата
            drawing = False                                                                                 # Флаг процесса рисования = False
            x1, x2 = min(ix, x), max(ix, x)                                                                 # Запоминаем начальное и конечное положения курсора
            y1, y2 = min(iy, y), max(iy, y)
            
            if (x2 - x1) > 3 and (y2 - y1) > 3:                                                             # Исключаем случайные клики без изменения размера
                template_coords.append([x1, y1, x2, y2])
                print(f"Добавлен прямоугольник {len(template_coords)}: [{x1}, {y1}, {x2}, {y2}]")         
            update_image()

    def update_image():                                                                                     # Функция обновления изображения с выделенными боксами
        nonlocal image_working                                                                              # Объявляем в начале
        image_working = image_original.copy()
        for rect in template_coords:
            cv2.rectangle(image_working, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        
        info_text = f"Boxes: {len(template_coords)}"                                                        # Показываем информацию о количестве боксов прямо на изображении
        if num_box is not None:
            info_text += f"/{num_box}"
        cv2.putText(image_working, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Selecting templates", image_working)

    
    cv2.namedWindow("Selecting templates", cv2.WINDOW_NORMAL)                                               # Создаем окно с возможностью изменения размера и настраиваем его размер
    cv2.resizeWindow("Selecting templates", 1200, 800)                                                      # (На моей системе возникала проблема с выводом этого изображения поэтому добавил этот кусок кода)
    cv2.imshow("Selecting templates", image_working)
    cv2.setMouseCallback("Selecting templates", draw_rectangle)
    
    print("\nИнструкция по разметке:")
    print("  - Зажмите ЛКМ и тяните мышку, чтобы выделить объект.")
    print("  - Нажмите BACKSPACE, чтобы удалить последнюю рамку.")
    print("  - Нажмите ENTER, чтобы завершить выбор.")
    
    if num_box is not None:
        print(f"  - Нужно выделить ровно {num_box} объект(ов).")
    
    while True:                                                                                             # Цикл для 
        key = cv2.waitKey(1) & 0xFF
        if key == 13:                                                                                       # Если нажат Enter, то запускаем проверку количества выделенных боксов
            n = len(template_coords)
            
            if num_box is not None and n != num_box:                                                        # Проверяем количество выделенных боксов
                print(f"\nОшибка: выделено {n} прямоугольников, а нужно {num_box}.")
                print("Исправьте количество и нажмите ENTER снова.")
                continue  # Не выходим из цикла, даем пользователю исправить
            elif n == 0:
                print("\nОшибка: не выделено ни одного прямоугольника.")
                print("Выделите хотя бы один объект.")
                continue
            else:
                break                                                                                       # В случае, если количество боксов не превышает допустимое, выходим из цикла
                
        elif key == 8:                                                                                      # Если нажат Backspace, то удаляем последний нарисованный прямоугольник
            if template_coords:
                removed = template_coords.pop()
                print(f"✗ Удален прямоугольник {len(template_coords) + 1}: {removed}")
                print(f"  Осталось прямоугольников: {len(template_coords)}")
                update_image()
            else:
                print("Нет прямоугольников для удаления")
    
    cv2.destroyAllWindows()
    
    print(f"\nИтого выделено примеров: {len(template_coords)}")
    return template_coords

def add_bottom_text_border(image, text, border_height=20,                                                   # Функция для добавления текста и рамки внизу изображения-результата
                           text_color=(0, 0, 0), 
                           bg_color=(255, 255, 255),
                           font_scale=0.7,
                           thickness=1):
    h, w = image.shape[:2]
    
    new_image = cv2.copyMakeBorder(                                                                         # Создаем новое изображение с дополнительной полосой снизу
        image, 
        0,              # top
        border_height,  # bottom
        0,              # left
        0,              # right
        cv2.BORDER_CONSTANT, 
        value=bg_color
    )
    
    font = cv2.FONT_HERSHEY_SIMPLEX                                                                         # Получаем размер текста для точного центрирования
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    text_x = (w - text_w) // 2                                                                              # Корректируем позицию для центрирования
    text_y = h + (border_height + text_h) // 2
    
    cv2.putText(new_image, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    return new_image

def main():

    print("Введите путь к изображению или 0 чтобы использовать пример")
    image_path = input().strip()
    
    if image_path == "0":
        print("Выбран пример")
        image_path = "images/image1.jpg"
    
    if not os.path.exists(image_path):                                                                       # Проверка существования файла
        print(f"Ошибка: файл {image_path} не найден!")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: не удалось считать изображение. Проверьте формат файла.")
        return
    
    print("\n=== РАЗМЕТКА ШАБЛОНА ===")
    print("Выделите объект, который будет использоваться как шаблон для поиска")
    
    template_boxes = get_box(image, num_box=1)
    
    if not template_boxes:
        print("Ошибка: шаблон не был выделен!")
        return
    
    template_box = template_boxes[0]
    print(f"✓ Шаблон выделен: {template_box}")
    
    # Вырезаем шаблон из изображения
    x1, y1, x2, y2 = template_box
    template = image[y1:y2, x1:x2]
    
    # Вызов классического алгоритма
    try:
        count, image_new = classic(image, template)
    except Exception as e:
        print(f"Ошибка при работе алгоритма: {e}")
        return
    
    text = f"{count} objects found"         
    image_with_border = add_bottom_text_border(image_new, text, border_height=60)
    
    os.makedirs("images", exist_ok=True)
    
    output_path = "images/result.jpg"
    cv2.imwrite(output_path, image_with_border)
    print(f"\nРезультат сохранён в {output_path}")
    print(f"Найдено объектов: {count}")
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 1200, 800)
    cv2.imshow("result", image_with_border)
    print("\nНажмите любую клавишу для закрытия окна...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()