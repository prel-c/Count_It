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
    Theta=np.zeros_like(image, dtype=np.float64)
        
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
            
            # Вычисляем величину градиента и направление
            M[i, j] = np.sqrt(gx**2 + gy**2)
            Theta[i, j]=np.atan2(gy, gx)
    
    return M, Theta

def NMS(M, Theta):
    Suppressed = np.zeros_like(M, dtype=np.float64)
    h, w = M.shape
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = Theta[i, j]
            angle = np.degrees(angle) % 180

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                n1 = M[i, j-1]
                n2 = M[i, j+1]
            elif (22.5 <= angle < 67.5):
                n1 = M[i-1, j+1]
                n2 = M[i+1, j-1]
            elif (67.5 <= angle < 112.5):
                n1 = M[i-1, j]
                n2 = M[i+1, j]
            else:
                n1 = M[i-1, j-1]
                n2 = M[i+1, j+1]
            
            if M[i, j] >= n1 and M[i, j] >= n2:
                Suppressed[i, j] = M[i, j]
    
    return Suppressed

def Double_threshold_filtering(Suppressed, bottom_treshold, top_treshold):
    Filtered=np.zeros_like(Suppressed, dtype=np.float64)
    for i in range(1, len(Suppressed)-1):
        for j in range(1, len(Suppressed[0])-1):
            if Suppressed[i, j]>top_treshold:
                Filtered[i, j]=255
            elif Suppressed[i, j]>bottom_treshold:
                Filtered[i, j]=100
    return Filtered

def normalizanion(M):
    return M/np.sum(M)*len(M)*len(M[0])


img = cv2.cvtColor((cv2.imread("image.jpg")), cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 41)

M, Theta = sobel(img)
cv2.imwrite('M.jpg', M)

Suppressed=NMS(M, Theta)
cv2.imwrite('Suppressed.jpg', Suppressed)

M=normalizanion(M)
Suppressed=NMS(M, Theta)



Filtered=Double_threshold_filtering(Suppressed, 1.5, 2)


cv2.imwrite('Filtered.jpg', Filtered)
