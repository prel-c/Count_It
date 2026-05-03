import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
____________________________________________________

Функции реализации метода LBP
____________________________________________________

"""

def LBP(i, j, photo):
    """
    Функция поиска локального бинарного шаблона (LBP) и перевод его в десятичную систему

    Ввод - координаты пикселя и матрица ч/б изображения
    Вывод - число в десятичной системе, соответсвующее LBP

    """
    st=[]
    lbp=0
    px = [(i-1, j),
          (i-1, j+1),
          (i, j+1),
          (i+1, j+1),
          (i+1, j),
          (i+1, j-1),
          (i, j-1),
          (i-1, j-1)]

    if 0<i<len(photo)-1 and 0<j<len(photo[0])-1:
        for i1, j1 in px:
            if photo[i1, j1]>photo[i, j]:
                st.append(1)
            else:
                st.append(0)
        for i in range (len(st)):
            lbp+=st[i]*2**(len(st)-1-i)
        return(lbp)
    else:
        return 0

def LBP_Matrix(photo):
    """
    Функция для создания матрицы из LBP

    Ввод - матрица ч/б изображения
    Вывод - матрица из LBP

    """
    LBP_img=np.full((len(photo), len(photo[0])), np.nan)
    for i in range (1, len(photo)-1):
        for j in range (1, len(photo[0])-1):
            LBP_img[i, j]=LBP(i, j, photo)
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
    if chi_square < 0.5:
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
    threshold = 0.5
    locations = np.where(result >= threshold)
    count = 0
    rectangles = []

    for pt in zip(*locations[::-1]):
        rectangles.append([pt[0], pt[1], w, h])

    rectangles = merge_duplicate_rectangles(rectangles, threshold=0.1)
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
    img=LBP_Matrix(img)
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
    Вывод - Матрица, содержащая в себе магнитуду каждого пикселя

    """ 

    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    
    h, w = image.shape
    
    M = np.zeros_like(image, dtype=np.float64)
        
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            region = image[i-1:i+2, j-1:j+2]
            gx = np.sum(region * Gx)
            gy = np.sum(region * Gy)
            M[i, j] = np.sqrt(gx**2 + gy**2)
    M = (M / M.max()) * 255
    M = M.astype(np.uint8)
    return M


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
image = cv2.GaussianBlur(image, (5, 5), 41)
Magnitude_im = sobel(image)

template = cv2.GaussianBlur(template, (5, 5), 41)
Magnitude_te = sobel(template)

# Цикл изменения масштаба изображения.
template_copy=np.copy(template)
all_rectangles = []

for i in range (0, len(template)*2//3, 1):

    # Именение масштаба изображения
    new_size = (len(template[0])-i, len(template)-i) 
    Magnitude_te = cv2.resize(Magnitude_te, new_size, interpolation=cv2.INTER_AREA)
    template_copy = cv2.resize(template_copy, new_size, interpolation=cv2.INTER_AREA)

    # Шаблонный поиск
    rectangles, count_template_search=template_search(Magnitude_im, Magnitude_te)
    all_rectangles.extend(rectangles)
    image_rec_t = drawing_rec(image_RGB_copy, rectangles)
    cv2.imwrite('t.jpg', image_rec_t)

    # Создание LBP гистограммы для шаблона
    LBP_img=LBP_Matrix(template_copy)
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
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image_rec_final, cv2.COLOR_BGR2RGB))  # Конвертируем BGR в RGB
plt.title('Final_Check')
plt.axis('off')
plt.show()
#cv2.imwrite('Final_Check.jpg', image_rec_final)
print("Финальное количество уникальных объектов:", len(rectangles_new))