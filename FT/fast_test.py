import cv2
import numpy as np
from scipy.ndimage import convolve
from sklearn.feature_extraction.image import extract_patches_2d

def test_optimized(image, template):
    """Оптимизированная версия"""
    
    def lbp_fast(img):
        """Векторизованный LBP"""
        h, w = img.shape
        lbp_image = np.zeros((h, w), dtype=np.uint8)
        
        # Создаем смещенные копии для векторизации
        img_pad = np.pad(img, 1, mode='constant')
        
        # Сравнение с центральным пикселем
        center = img
        neighbors = [
            img_pad[:-2, 1:-1],   # вверх
            img_pad[:-2, 2:],     # вверх-вправо
            img_pad[1:-1, 2:],    # вправо
            img_pad[2:, 2:],      # вниз-вправо
            img_pad[2:, 1:-1],    # вниз
            img_pad[2:, :-2],     # вниз-влево
            img_pad[1:-1, :-2],   # влево
            img_pad[:-2, :-2]     # вверх-влево
        ]
        
        # Векторизованное сравнение
        for i, n in enumerate(neighbors):
            lbp_image += (n > center).astype(np.uint8) << i
            
        return lbp_image[1:-1, 1:-1]  # Убираем границы
    
    def lbp_histogram_normalized(img):
        """Быстрая гистограмма LBP"""
        lbp = lbp_fast(img)
        hist = np.bincount(lbp.ravel(), minlength=256)
        return hist / (hist.sum() + 1e-10)
    
    def sobel_fast(img):
        """Векторизованный Sobel"""
        # Ядра Собеля
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Конволюция через scipy (быстрее самодельных циклов)
        gx = convolve(img.astype(np.float32), Gx, mode='constant')
        gy = convolve(img.astype(np.float32), Gy, mode='constant')
        
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = (magnitude / (magnitude.max() + 1e-10)) * 255
        return magnitude.astype(np.uint8)
    
    def template_search_multi_scale(magnitude_im, template_magnitude, scales, threshold=0.6):
        """Поиск на нескольких масштабах без повторной обработки"""
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
    hist_template = lbp_histogram_normalized(template_blur)
    
    valid_rectangles = []
    for rect in rectangles:
        x, y, w, h = map(int, rect)
        roi = image_blur[y:y+h, x:x+w]
        if roi.size > 0:
            hist_roi = lbp_histogram_normalized(roi)
            # Вычисляем chi-square
            chi2 = np.sum((hist_roi - hist_template)**2 / (hist_roi + hist_template + 1e-10))
            if chi2 < 0.3:
                valid_rectangles.append(rect)
    
    return len(valid_rectangles)