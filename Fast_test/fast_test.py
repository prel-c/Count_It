import cv2
import numpy as np
from scipy.ndimage import convolve


def test_optimized(image, template):
    """Оптимизированная версия"""
    
    def lbp_fast(img):

        """
        Функция поиска локального бинарного шаблона (LBP) и перевод его в десятичную систему

        Ввод - матрица ч/б изображения
        Вывод - матрица из LBP чисел

        """

        h, w = img.shape                                                            # Получаем размер изображения
        lbp_image = np.zeros((h, w), dtype=np.uint8)                                # Создаём матрицу для заполнения LBP числами
        img_pad = np.pad(img, 1, mode='constant')                                   # Увеличиваем матрицу до размеров i+2, j+2

        neighbors = [                                                               # Создаём матрицы такого же размера как и img, но сдвинутые в ту сторону относительно которой мы сравниваем
            img_pad[:-2, 1:-1],                                                         # север      
            img_pad[:-2, 2:],                                                           # северо-восток
            img_pad[1:-1, 2:],                                                          # восток
            img_pad[2:, 2:],                                                            # юго-восток
            img_pad[2:, 1:-1],                                                          # юг
            img_pad[2:, :-2],                                                           # юго-запад
            img_pad[1:-1, :-2],                                                         # запад
            img_pad[:-2, :-2]                                                           # северо-запад
        ]           

        for i, n in enumerate(neighbors):                                           # Векторизованное сравнение
            lbp_image += (n > img).astype(np.uint8) << i            

        return lbp_image[1:-1, 1:-1]                                                # Возвращаем матрицу из LBP чисел, обрезанную до размера исходного изображения 

    def lbp_histogram_normalized(lbp):
        """
        Функция для создания гистограммы из матрицы LBP

        Ввод - LBP матрица
        Вывод - Список размером 256 позиций. 
                В каждой позиции записано количество всех чисел соответсвующих позиции в списке.

        """                                        
        hist = np.bincount(lbp.ravel(), minlength=256)                              # Преобразуем массив в строку и считаем сколько раз повторяется каждый элемент значением от 0 до 255
        return hist / (hist.sum() + 1e-10)                                          # Возвращаем L1 нормализованную гистограмму
    
    def sobel_fast(img):
        """
        Функция применения ядра собеля к ч/б изображению

        Ввод - Матрица, соответсвующая ч/б изображению
        Вывод - Матрица, содержащая в себе магнитуду каждого пикселя

        """ 
        h, w = img.shape
        Gx = np.array([[-1, 0, 1],                                                  # Создаём ядра собеля
                       [-2, 0, 2], 
                       [-1, 0, 1]], 
                       dtype=np.float32)
        
        Gy = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], 
                       dtype=np.float32)
        
        img_pad = np.pad(img, 1, mode='constant')                                   # Увеличиваем матрицу до размеров i+2, j+2
        gx=(Gx[0,0]*img_pad[0:h, 0:w]+                                                  #-1*северо-запад
            Gx[0,1]*img_pad[0:h, 1:w+1]+                                                # 0*север
            Gx[0,2]*img_pad[0:h, 2:w+2]+                                                # 1*северо-восток
            Gx[1,0]*img_pad[1:h+1, 0:w]+                                                #-2*запад
            Gx[1,1]*img_pad[1:h+1, 1:w+1]+                                              # 0*центр
            Gx[1,2]*img_pad[1:h+1, 2:w+2]+                                              # 2*восток
            Gx[2,0]*img_pad[2:h+2, 0:w]+                                                #-1*юго-запад
            Gx[2,1]*img_pad[2:h+2, 1:w+1]+                                              # 0*юг
            Gx[2,2]*img_pad[2:h+2, 2:w+2])                                              # 1*юго-восток

        gy=(Gy[0,0]*img_pad[0:h, 0:w]+                                                  #-1*северо-запад
            Gy[0,1]*img_pad[0:h, 1:w+1]+                                                # 0*север
            Gy[0,2]*img_pad[0:h, 2:w+2]+                                                # 1*северо-восток
            Gy[1,0]*img_pad[1:h+1, 0:w]+                                                #-2*запад
            Gy[1,1]*img_pad[1:h+1, 1:w+1]+                                              # 0*центр
            Gy[1,2]*img_pad[1:h+1, 2:w+2]+                                              # 2*восток
            Gy[2,0]*img_pad[2:h+2, 0:w]+                                                #-1*юго-западs
            Gy[2,1]*img_pad[2:h+2, 1:w+1]+                                              # 0*юг
            Gy[2,2]*img_pad[2:h+2, 2:w+2])                                              # 1*юго-вос      
        
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = (magnitude / (magnitude.max() + 1e-10)) * 255                       
        return magnitude.astype(np.uint8)
    
    def template_search_multi_scale(magnitude_im, template_magnitude, scales, threshold=0.6):
        """
        Функция реализации шаблонного поиска

        Ввод - Матрица, соответсвующая ч/б изображению
        Вывод - Массив с координатами прямоугольников, соответсвующие потенциальным областям с объектом.
                Число таких областей

        """                      
        all_rectangles = []
        h_im, w_im = magnitude_im.shape
        h_tmpl, w_tmpl = template_magnitude.shape
        
        for scale in scales:
            new_w = max(10, int(w_tmpl * scale))
            new_h = max(10, int(h_tmpl * scale))
            
            if new_w > w_im or new_h > h_im:
                continue
                
            tmpl_scaled = cv2.resize(template_magnitude, (new_w, new_h))
            result = cv2.matchTemplate(magnitude_im, tmpl_scaled, cv2.TM_CCOEFF_NORMED)
            
            ys, xs = np.where(result >= threshold)
            
            for x, y in zip(xs, ys):
                all_rectangles.append([x, y, new_w, new_h])
        
        return all_rectangles
    
    def merge_rectangles_fast(rectangles, iou_threshold=0.3):
        """Оптимизированное удаление дубликатов"""
        if len(rectangles) <= 1:
            return rectangles
        
        # Преобразуем в numpy массив
        rects = np.array(rectangles, dtype=np.float32)
        
        # Координаты углов
        x1 = rects[:, 0]
        y1 = rects[:, 1]
        x2 = rects[:, 0] + rects[:, 2]
        y2 = rects[:, 1] + rects[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(areas)[::-1]
        
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(rectangles[i])
            
            if len(idxs) == 1:
                break
            
            # Вычисляем IoU
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[idxs[1:]] - intersection)
            
            idxs = idxs[1:][iou < iou_threshold]
        
        return keep
    
    # Основная логика
    # Предобработка один раз
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 41)
    template_blur = cv2.GaussianBlur(template_gray, (5, 5), 41)
    
    magnitude_im = sobel_fast(image_blur)
    magnitude_tmpl = sobel_fast(template_blur)
    
    # Создаем масштабы
    scales = np.linspace(0.5, 1.5, 10)  # Уменьшил количество масштабов
    
    # Поиск шаблонов
    rectangles = template_search_multi_scale(magnitude_im, magnitude_tmpl, scales, threshold=0.4)
    rectangles = merge_rectangles_fast(rectangles, iou_threshold=0.3)
    
    # LBP проверка (только для найденных прямоугольников)
    lbp_template = lbp_fast(template_blur)
    hist_template = lbp_histogram_normalized(lbp_template)
    
    valid_rectangles = []
    for rect in rectangles:
        x, y, w, h = map(int, rect)
        roi = image_blur[y:y+h, x:x+w]
        if roi.size > 0:
            lbp_roi=lbp_fast(roi)
            hist_roi = lbp_histogram_normalized(lbp_roi)
            # Вычисляем chi-square
            chi2 = np.sum((hist_roi - hist_template)**2 / (hist_roi + hist_template + 1e-10))
            if chi2 < 0.3:
                valid_rectangles.append(rect)
    
    return len(valid_rectangles)