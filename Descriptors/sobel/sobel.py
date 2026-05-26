import cv2
import numpy as np
import matplotlib.pyplot as plt

def Sobel(img):
    """
    Функция применения ядра собеля к ч/б изображению
    Ввод - Матрица, соответсвующая ч/б изображению
    Вывод - Матрица, содержащая в себе магнитуду каждого пикселя
    """ 
    h, w=img.shape
    Gx=np.array([[-1, 0, 1],                                                    # Создаём ядра собеля
                 [-2, 0, 2], 
                 [-1, 0, 1]], 
                 dtype=np.float32)
    
    Gy=np.array([[-1, -2, -1], 
                 [0, 0, 0], 
                 [1, 2, 1]], 
                 dtype=np.float32)
    
    img_new_size=np.pad(img, 1, mode='constant')                                # Увеличиваем матрицу до размеров i+2, j+2
    gx=(Gx[0,0]*img_new_size[0:h, 0:w]+                                             #-1*северо-запад
        Gx[0,2]*img_new_size[0:h, 2:w+2]+                                           # 1*северо-восток
        Gx[1,0]*img_new_size[1:h+1, 0:w]+                                           #-2*запад
        Gx[1,2]*img_new_size[1:h+1, 2:w+2]+                                         # 2*восток
        Gx[2,0]*img_new_size[2:h+2, 0:w]+                                           #-1*юго-запад
        Gx[2,2]*img_new_size[2:h+2, 2:w+2])                                         # 1*юго-восток
    gy=(Gy[0,0]*img_new_size[0:h, 0:w]+                                             #-1*северо-запад
        Gy[0,1]*img_new_size[0:h, 1:w+1]+                                           #-2*север
        Gy[0,2]*img_new_size[0:h, 2:w+2]+                                           #-1*северо-восток
        Gy[2,0]*img_new_size[2:h+2, 0:w]+                                           # 1*юго-запад
        Gy[2,1]*img_new_size[2:h+2, 1:w+1]+                                         #-1*юг
        Gy[2,2]*img_new_size[2:h+2, 2:w+2])                                         # 1*юго-восток      
    magnitude=np.sqrt(gx**2 + gy**2)                                            # Считаем магнитуду
    magnitude=(magnitude/(magnitude.max()+1e-10))*255                           # Нормализуем по максимальному значению
    cv2.imwrite('X.jpg', gx)
    cv2.imwrite('Y.jpg', gy)
    return magnitude.astype(np.uint8)

img = cv2.cvtColor((cv2.imread("image.png")), cv2.COLOR_BGR2GRAY)
obj=cv2.cvtColor((cv2.imread("image1.png")), cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (5, 5), 41)

M = Sobel(img)
#M1=ad_cont(M, max(len(obj), len(obj[0])))

cv2.imwrite('M.jpg', M)
#cv2.imwrite("M1.jpg", M1)

