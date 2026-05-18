import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel(image):

    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    
    # Получаем размеры изображения
    h, w = image.shape
    
    # Создаем массивы для результатов
    X = np.zeros_like(image, dtype=np.float64)
    Y = np.zeros_like(image, dtype=np.float64)
    M = np.zeros_like(image, dtype=np.float64)
        
    # Применяем свертку (обходим пиксели, кроме границ)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Вырезаем область 3x3 вокруг текущего пикселя
            region = image[i-1:i+2, j-1:j+2]
            
            # Вычисляем градиент по X и Y
            gx = np.sum(region * Gx)
            gy = np.sum(region * Gy)
            
            X[i, j] = gx
            Y[i, j] = gy
            
            # Вычисляем величину градиента
            M[i, j] = np.sqrt(gx**2 + gy**2)

    # Нормализуем результат для отображения (0-255)
    M = (M / M.max()) * 255
    M = M.astype(np.uint8)
    
    return M

"""
def detect_contours_adaptive(M):

    
    # Автоматический порог OTSU
    _, binary = cv2.threshold(M, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Морфологические операции для улучшения
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Закрытие дырок
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Удаление шума
    
    # Находим контуры
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтруем контуры по площади
    min_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Рисуем результат
    result = img.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    return binary, result
"""

def ad_cont(M, a):
    M1 = np.zeros_like(M, dtype=np.float64)
    for i in range (0, len(M)-a, a):
        for j in range (0, len(M[0])-a, a):
            m=M[i:i+a, j:j+a]
            p=np.sum(m)/len(m)/len(m[0])
            for i1 in range(a):
                for j1 in range(a):
                    if M[i+i1, j+j1]>p:
                        M1[i+i1, j+j1]=255
    return M1

img = cv2.cvtColor((cv2.imread("image.png")), cv2.COLOR_BGR2GRAY)
obj=cv2.cvtColor((cv2.imread("image1.png")), cv2.COLOR_BGR2GRAY)
"""
for i in range (3,16,2):
    for j in range (1, 100, 5):

        img = cv2.GaussianBlur(img, (i, i), j)

        M = sobel(img)
        M1=ad_cont(M, max(len(obj), len(obj[0])))

        cv2.imwrite(f"/mnt/c/CV/COD/Res/M1_{i}_{j}.jpg", M1)

"""

img = cv2.GaussianBlur(img, (5, 5), 41)

M = sobel(img)
M1=ad_cont(M, max(len(obj), len(obj[0])))

cv2.imwrite('M.jpg', M)
cv2.imwrite("M1.jpg", M1)



