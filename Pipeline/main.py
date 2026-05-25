import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import warnings
warnings.filterwarnings("ignore")
import cv2
from classic import classic


def get_box(image, num_box=None):
    """Функция ручной разметки боксов"""
    image_original = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_working = image_original.copy()
    
    template_coords = []  # Список для хранения [x1, y1, x2, y2]
    drawing = False        # Флаг процесса рисования
    ix, iy = -1, -1       # Стартовые координаты мыши

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, template_coords, image_working
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = image_original.copy()
                # Рисуем все ранее сохраненные прямоугольники
                for rect in template_coords:
                    cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                # Рисуем рамку, которую тянем прямо сейчас (синим цветом для отличия)
                cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
                cv2.imshow("Selecting templates", img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, x2 = min(ix, x), max(ix, x)
            y1, y2 = min(iy, y), max(iy, y)
            
            # Исключаем случайные клики без изменения размера
            if (x2 - x1) > 3 and (y2 - y1) > 3:
                template_coords.append([x1, y1, x2, y2])
                print(f"✓ Добавлен прямоугольник {len(template_coords)}: [{x1}, {y1}, {x2}, {y2}]")
            update_image()

    def update_image():
        nonlocal image_working
        image_working = image_original.copy()
        for rect in template_coords:
            cv2.rectangle(image_working, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        cv2.imshow("Selecting templates", image_working)

    cv2.imshow("Selecting templates", image_working)
    cv2.setMouseCallback("Selecting templates", draw_rectangle)
    
    print("\nИнструкция по разметке:")
    print("  - Зажмите ЛКМ и тяните мышку, чтобы выделить объект.")
    print("  - Нажмите BACKSPACE, чтобы удалить последнюю рамку.")
    print("  - Нажмите ENTER, чтобы завершить выбор и запустить подсчет.\n")

    if num_box != 1:
        print(f'Выделите ровно 1 бокс!!!')
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        elif key == 8:  # Backspace
            if template_coords:
                removed = template_coords.pop()
                print(f"✗ Удален прямоугольник {len(template_coords) + 1}: {removed}")
                print(f"  Осталось прямоугольников: {len(template_coords)}")
                update_image()
            else:
                print("Нет прямоугольников для удаления")
                
    cv2.destroyAllWindows()

    n = len(template_coords)

    if num_box != None and n != num_box:
        raise ValueError("Неккоректное число боксов!")

    print(f"\nИтого выделено примеров: {n}")
    return template_coords


count, image_new=classic(image)


if __name__ == "__main__":

    print("Введите путь к изображению или 0 чтобы использовать пример")
    image_path = input()
    if image_path == "0":
        print("Выбран пример")
        image_path = "image.jpg"

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Не удалось считать изображение")

