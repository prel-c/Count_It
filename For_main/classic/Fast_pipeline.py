import cv2
import numpy as np

"""
____________________________________________________

Функции реализации метода LBP
____________________________________________________

"""
def LBP_Matrix_fast(photo):
    """Быстрая версия создания LBP матрицы"""
    h, w = photo.shape
    LBP_img = np.zeros((h, w), dtype=np.uint8)
    
    # Предварительно вычисляем пороговые значения
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = photo[i, j]
            # Векторизованная проверка соседей
            code = 0
            code |= (photo[i-1, j] > center) << 7
            code |= (photo[i-1, j+1] > center) << 6
            code |= (photo[i, j+1] > center) << 5
            code |= (photo[i+1, j+1] > center) << 4
            code |= (photo[i+1, j] > center) << 3
            code |= (photo[i+1, j-1] > center) << 2
            code |= (photo[i, j-1] > center) << 1
            code |= (photo[i-1, j-1] > center) << 0
            LBP_img[i, j] = code
    return LBP_img


def LBP_Ghist(LBP_img):
    """
    Функция для создания гистограммы из матрицы LBP

    Ввод - LBP матрица
    Вывод - Список размером 256 позиций. 
            В каждой позиции записано количество всех чисел соответсвующих позиции в списке.

    """
    ghistogramm=np.zeros(256)
    for i in range(256):
        ghistogramm[i]=np.count_nonzero(LBP_img == i)
    return ghistogramm

def LBP_Ghist_normalization(ghistogramm):
    """
    Функция нормализации гистограммы.

    Ввод - Список 256 позиций (гистограмма)
    Вывод - Нормализованный список 256 позиций (гистограмма)

    """
    return ghistogramm/np.sum(ghistogramm)

def chi_square(hist1, hist2, epsilon=1e-10):

    """
    Функция проверки связи между гистограммами.

    Ввод - два списока на 256 позиций (гистограммы)
    Вывод - Истина/ Ложь

    """
    chi_square=np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon))
    if chi_square < 0.4:
        return True
    else:
        return False



"""
____________________________________________________

Остальные функции
____________________________________________________

"""

def template_search(image, template):

    """
    Функция реализации шаблонного поиска

    Ввод - Матрица, соответсвующая ч/б изображению
    Вывод - Массив с координатами прямоугольников, соответсвующие потенциальным областям с объектом.
            Число таких областей

    """                      

    w, h = template.shape[::-1]

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.17
    locations = np.where(result >= threshold)
    count = 0
    rectangles = []

    for pt in zip(*locations[::-1]):
        rectangles.append([pt[0], pt[1], w, h])

    rectangles = merge_duplicate_rectangles(rectangles, threshold=0.4)
    count=len(rectangles)
    return rectangles, count

def drawing_rec(image, rectangles):

    """
    Функция рисования прямоугольников

    Ввод - Матрица, соответсвующая цветному изображению и координаты прямоугольников
    Вывод - Матрица, соответсвующая цветному изображению с нарисованными прямоугольниками

    """  

    for (x, y, w, h) in rectangles:
        cv2.rectangle(image, (x,y), (x+w,y+h), 150, 2)
    return image

def cheking(image, rectangle, ghist1):

    """
    Функция проверки методом LBP областей найденных с помощью шаблонного поиска

    Ввод - Матрица, соответсвующая ч/б изображению, координаты прямоугольника, гистограмма шаблона
    Вывод - Истина/Ложь

    """  
    
    x=rectangle[0]
    y=rectangle[1]
    w=rectangle[2]
    h=rectangle[3]
    img=image[y:y+h, x:x+w]
    img=LBP_Matrix_fast(img)
    ghist=LBP_Ghist_normalization(LBP_Ghist(img))
    if chi_square(ghist, ghist1, epsilon=1e-10) is True:
        return True
    else:
        return False

def merge_duplicate_rectangles(rectangles, threshold=0.1):

    """
    Функция удаления прямоугольников, дублирующих существующий

    Ввод - массив с набором координат прямоугольников
    Вывод - масив с набором координат "уникальных" прямоугольников

    """ 

    if len(rectangles) <= 1:
        return rectangles
    borders = []
    for (x, y, w, h) in rectangles:
        borders.append([x, y, x + w, y + h])
    borders = np.array(borders)
    S = (borders[:, 2] - borders[:, 0]) * (borders[:, 3] - borders[:, 1])
    S_sorted = np.argsort(S)[::-1]

    keep = []
    while len(S_sorted) > 0:
        current = S_sorted[0]
        keep.append(rectangles[current])
        if len(S_sorted) == 1:
            break

        remaining = S_sorted[1:]
        
        x1 = np.maximum(borders[current, 0], borders[remaining, 0])
        y1 = np.maximum(borders[current, 1], borders[remaining, 1])
        x2 = np.minimum(borders[current, 2], borders[remaining, 2])
        y2 = np.minimum(borders[current, 3], borders[remaining, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_current = S[current]
        area_remaining = S[remaining]
        iou = intersection / (area_current + area_remaining - intersection)
        
        S_sorted = remaining[iou < threshold]
    
    return keep

"""
____________________________________________________

Функции предобработки изображения
____________________________________________________

"""

def sobel(image):

    """
    Функция применения ядра собеля к ч/б изображению

    Ввод - Матрица, соответсвующая ч/б изображению
    Вывод - Матрицы, содержащая в себе магнитуду каждого пикселя и направление градиента

    """ 

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
    Filtered=np.zeros_like(Suppressed, dtype=np.uint8)
    for i in range(1, len(Suppressed)-1):
        for j in range(1, len(Suppressed[0])-1):
            if Suppressed[i, j]>top_treshold:
                Filtered[i, j]=255
            elif Suppressed[i, j]>bottom_treshold:
                Filtered[i, j]=100
    return Filtered

def normalizanion(M):
    return M/np.sum(M)*len(M)*len(M[0])


"""
____________________________________________________

Конец раздела функций
____________________________________________________

"""

# Ввод изображения и обьекта
image_RGB = cv2.imread("image.jpg")
image_RGB_copy=np.copy(image_RGB)
template_RGB = cv2.imread("image1.jpg")

image=cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
template=cv2.cvtColor(template_RGB, cv2.COLOR_BGR2GRAY)

# Предобработка изображения и шаблона соответсвенно (Гауссово размытие и ядро собеля)
image = cv2.GaussianBlur(image, (5, 5), 1)
M_im, Theta_im = sobel(image)
M_im=normalizanion(M_im)
Suppressed_im=NMS(M_im, Theta_im)
Filtered_im=Double_threshold_filtering(Suppressed_im, 1.5, 2)
cv2.imwrite('Filtered_im.jpg', Filtered_im)


template = cv2.GaussianBlur(template, (5, 5), 41)
M_te, Theta_te= sobel(template)
M_te=normalizanion(M_te)
Suppressed_te=NMS(M_te, Theta_te)
Filtered_te=Double_threshold_filtering(Suppressed_te, 1.5, 2)
cv2.imwrite('Filtered_te.jpg', Filtered_te)

# Цикл изменения масштаба изображения.
template_copy=np.copy(template)
all_rectangles = []

for i in range (0, len(template)*2//3, 2):

    # Именение масштаба изображения
    new_size = (len(template[0])-i, len(template)-i) 
    Filtered_te = cv2.resize(Filtered_te, new_size, interpolation=cv2.INTER_AREA)
    template_copy = cv2.resize(template_copy, new_size, interpolation=cv2.INTER_AREA)

    # Шаблонный поиск
    rectangles, count_template_search=template_search(Filtered_im, Filtered_te)
    all_rectangles.extend(rectangles)
    image_rec_t = drawing_rec(image_RGB_copy, rectangles)
    cv2.imwrite('t.jpg', image_rec_t)

    # Создание LBP гистограммы для шаблона
    LBP_img=LBP_Matrix_fast(template_copy)
    ghist1=LBP_Ghist_normalization(LBP_Ghist(LBP_img))

    # Удаление лишних прямоугольников
    all_rectangles = merge_duplicate_rectangles(all_rectangles)

    # Проверка с помощью метода LBP 
    rectangles_new=[]
    for rectangle in all_rectangles: 
        check = cheking(image, rectangle, ghist1)
        if check is True:
            rectangles_new.append(rectangle)

image_rec_final = drawing_rec(image_RGB, rectangles_new)
cv2.imwrite('Final_Check.jpg', image_rec_final)
print("Финальное количество уникальных объектов:", len(rectangles_new))

