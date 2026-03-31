import cv2
import numpy as np

"""
____________________________________________________

Функции реализации метода LBP
____________________________________________________

"""
# Поиск LBP
def LBP(i, j, photo):
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

# Составление матрицы LBP
def LBP_Matrix(photo):
    LBP_img=np.full((len(photo), len(photo[0])), np.nan)
    for i in range (1, len(photo)-1):
        for j in range (1, len(photo[0])-1):
            LBP_img[i, j]=LBP(i, j, photo)
    return LBP_img

# Состевление гистограммы для объекта
def LBP_Ghist(LBP_img):
    ghistogramm=np.zeros(256)
    for i in range(256):
        ghistogramm[i]=np.count_nonzero(LBP_img == i)
    return ghistogramm

# Нормализация гистограммы объекта
def LBP_Ghist_normalization(ghistogramm):
    return ghistogramm/np.sum(ghistogramm)

# Сраванение гистограмм
def chi_square(hist1, hist2, epsilon=1e-10):
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon))

# Функция для получения всех равномерных паттернов
def uniform_patterns():
    """Генерирует список равномерных LBP паттернов (59 значений)"""
    patterns = []
    # Равномерные паттерны (не более 2 переходов 0-1)
    for i in range(256):
        binary = format(i, '08b')
        transitions = 0
        for j in range(8):
            if binary[j] != binary[(j+1) % 8]:
                transitions += 1
        if transitions <= 2:
            patterns.append(i)
    return patterns

# Поиск равномерного LBP
def LBP_uniform(i, j, photo, uniform_patterns_list):
    """Вычисляет равномерный LBP код"""
    st = []
    px = [(i-1, j),
          (i-1, j+1),
          (i, j+1),
          (i+1, j+1),
          (i+1, j),
          (i+1, j-1),
          (i, j-1),
          (i-1, j-1)]

    if 0 < i < len(photo)-1 and 0 < j < len(photo[0])-1:
        for i1, j1 in px:
            if photo[i1, j1] > photo[i, j]:
                st.append(1)
            else:
                st.append(0)
        
        # Вычисляем количество переходов
        transitions = 0
        for k in range(8):
            if st[k] != st[(k+1) % 8]:
                transitions += 1
        
        # Если паттерн равномерный, возвращаем его значение
        if transitions <= 2:
            lbp = 0
            for k in range(len(st)):
                lbp += st[k] * 2**(len(st)-1-k)
            # Индексируем равномерные паттерны от 0 до 57
            if lbp in uniform_patterns_list:
                return uniform_patterns_list.index(lbp)
        else:
            # Неравномерные паттерны отправляем в 58-ю bin
            return 58
    else:
        return 0

# Составление матрицы равномерного LBP
def LBP_Uniform_Matrix(photo):
    """Создает матрицу равномерного LBP (59 значений)"""
    uniform_list = uniform_patterns()
    LBP_img = np.full((len(photo), len(photo[0])), np.nan)
    for i in range(1, len(photo)-1):
        for j in range(1, len(photo[0])-1):
            LBP_img[i, j] = LBP_uniform(i, j, photo, uniform_list)
    return LBP_img

# Составление гистограммы для равномерного LBP
def LBP_Uniform_Ghist(LBP_img):
    """Составляет гистограмму для равномерного LBP (59 bins)"""
    ghistogramm = np.zeros(59)
    # Игнорируем nan значения (границы)
    valid_values = LBP_img[~np.isnan(LBP_img)]
    for i in range(59):
        ghistogramm[i] = np.count_nonzero(valid_values == i)
    return ghistogramm

"""
____________________________________________________

Остальные функции
____________________________________________________

"""

def template_search(image, template):                               

    w, h = template.shape[::-1]

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    locations = np.where(result >= threshold)
    count = 0
    rectangles = []

    for pt in zip(*locations[::-1]):
        rectangles.append([pt[0], pt[1], w, h])
        rectangles.append([pt[0], pt[1], w, h])

    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.5)
    count=len(rectangles)
    return rectangles, count


def drawing_rec(image, rectangles):
    for (x, y, w, h) in rectangles:
        cv2.rectangle(image, (x,y), (x+w,y+h), 150, 2)
    return image


def cheking(image, rectangle, ghist1):
    x=rectangle[0]
    y=rectangle[1]
    w=rectangle[2]
    h=rectangle[3]
    img=image[y:y+h, x:x+w]
    # Используем равномерный LBP
    img=LBP_Uniform_Matrix(img)
    ghist=LBP_Ghist_normalization(LBP_Uniform_Ghist(img))
    if chi_square(ghist, ghist1, epsilon=1e-10) < 0.15:
        return True, rectangle
    else:
        return False, rectangle
    
"""
____________________________________________________

Функции предобработки изображения
____________________________________________________

"""
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
    M = np.zeros_like(image, dtype=np.float64)
        
    # Применяем свертку (обходим пиксели, кроме границ)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Вырезаем область 3x3 вокруг текущего пикселя
            region = image[i-1:i+2, j-1:j+2]
            
            # Вычисляем градиент по X и Y
            gx = np.sum(region * Gx)
            gy = np.sum(region * Gy)

            # Вычисляем величину градиента
            M[i, j] = np.sqrt(gx**2 + gy**2)

    # Нормализуем результат для отображения (0-255)
    M = (M / M.max()) * 255
    M = M.astype(np.uint8)
    
    return M

def ad_cont(M, a):
    h, w = M.shape
    M1 = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h - a + 1, a):
        for j in range(0, w - a + 1, a):
            # Выделяем блок
            block = M[i:i+a, j:j+a]
            # Вычисляем среднее
            p = np.mean(block)
            # Сравниваем и устанавливаем значения
            M1[i:i+a, j:j+a] = (block > p).astype(np.uint8) * 255
    return M1


"""
____________________________________________________

Конец раздела функций
____________________________________________________

"""

# Ввод изображения и обьекта
image_RGB = cv2.imread("image.jpg")
image_RGB_copy=np.copy(image_RGB)
template_RGB = cv2.imread("image1.jpg")
itemplate_RGB_copy=np.copy(template_RGB)

image=cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
template=cv2.cvtColor(template_RGB, cv2.COLOR_BGR2GRAY)

# Предобработка
image = cv2.GaussianBlur(image, (5, 5), 41)
Magnitude_im = sobel(image)
cv2.imwrite('Magnitude_im.jpg', Magnitude_im)
Borders_im=ad_cont(Magnitude_im, max(len(template), len(template[0])))
cv2.imwrite('Borders_im.jpg', Borders_im)

template = cv2.GaussianBlur(template, (5, 5), 41)
Magnitude_te = sobel(template)
cv2.imwrite('Magnitude_te.jpg', Magnitude_te)
Borders_te=ad_cont(Magnitude_te, max(len(template), len(template[0])))
cv2.imwrite('Borders_te.jpg', Borders_te)

# Шаблонный поиск для изображения
rectangles, count_template_search=template_search(Magnitude_im, Magnitude_te)
image_rec_temp = drawing_rec(image_RGB_copy, rectangles)
print("Найдено объектов после шаблонного поиска:", count_template_search)
cv2.imwrite('Template_search.jpg', image_rec_temp)

# Поиск гистограммы объекта с использованием равномерного LBP
LBP_img = LBP_Uniform_Matrix(template)
ghist1 = LBP_Ghist_normalization(LBP_Uniform_Ghist(LBP_img))

# Проверка шаблонного поиска
rectangles_new=[]
count_after_cheking = 0
for rectangle in rectangles:
    check, rectangle = cheking(image, rectangle, ghist1)
    if check is True:
        count_after_cheking += 1
        rectangles_new.append(rectangle)
image_rec_check = drawing_rec(image_RGB, rectangles_new)

print("Найдено объектов после проверки:", count_after_cheking)
cv2.imwrite('Check.jpg', image_rec_check)