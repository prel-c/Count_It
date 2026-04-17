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

"""
____________________________________________________

Остальные функции
____________________________________________________

"""

# Шаблонный поиск
def template_search(image, template):                               

    w, h = template.shape[::-1]

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    locations = np.where(result >= threshold)
    count = 0
    rectangles = []

    for pt in zip(*locations[::-1]):
        rectangles.append([pt[0], pt[1], w, h])

    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.5)
    count=len(rectangles)
    return rectangles, count

# Рисование рамки
def drawing_rec(image, rectangles):
    for (x, y, w, h) in rectangles:
        cv2.rectangle(image, (x,y), (x+w,y+h), 150, 2)
    return image

# Проверка методом LBP
def cheking(image, rectangle, ghist1):
    x=rectangle[0]
    y=rectangle[1]
    w=rectangle[2]
    h=rectangle[3]
    img=image[y:y+h, x:x+w]
    img=LBP_Matrix(img)
    ghist=LBP_Ghist_normalization(LBP_Ghist(img))
    if chi_square(ghist, ghist1, epsilon=1e-10) < 0.4:
        return True, rectangle
    else:
        return False, rectangle

# Удаление дублирующихся прямоугольников
def merge_duplicate_rectangles(rectangles, threshold=0.3):
    if len(rectangles) <= 1:
        return rectangles
    boxes = []
    for (x, y, w, h) in rectangles:
        boxes.append([x, y, x + w, y + h])
    boxes = np.array(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_idx = np.argsort(areas)[::-1]
    keep = []
    while len(sorted_idx) > 0:
        current = sorted_idx[0]
        keep.append(rectangles[current])
        if len(sorted_idx) == 1:
            break

        remaining = sorted_idx[1:]
        
        x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
        y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_current = areas[current]
        area_remaining = areas[remaining]
        iou = intersection / (area_current + area_remaining - intersection)
        
        sorted_idx = remaining[iou < threshold]
    
    return keep

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

def ad_cont(M, a):
    h, w = M.shape
    M1 = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h - a + 1, a):
        for j in range(0, w - a + 1, a):
            block = M[i:i+a, j:j+a]
            p = np.mean(block)
            M1[i:i+a, j:j+a] = (block > p).astype(np.uint8) * 255
    return M1


"""
____________________________________________________

Конец раздела функций
____________________________________________________

"""

### Ввод изображения и обьекта
image_RGB = cv2.imread("image.jpg")
image_RGB_copy=np.copy(image_RGB)
template_RGB = cv2.imread("image1.jpg")

image=cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
template=cv2.cvtColor(template_RGB, cv2.COLOR_BGR2GRAY)

# Предобработка изображения и шаблона соответсвенно (Гауссово размытие и ядро собеля с адаптивным выделением контуров)
image = cv2.GaussianBlur(image, (5, 5), 41)
Magnitude_im = sobel(image)
Borders_im=ad_cont(Magnitude_im, max(len(template), len(template[0])))

template = cv2.GaussianBlur(template, (5, 5), 41)
Magnitude_te = sobel(template)
Borders_te=ad_cont(Magnitude_te, max(len(template), len(template[0])))

# Цикл изменения масштаба изображения.
template_copy=np.copy(template)
all_rectangles = []

for i in range (0, len(template)*2//3, 2):
    new_size = (len(template[0])-i, len(template)-i)  # (ширина, высота)
    Magnitude_te = cv2.resize(Magnitude_te, new_size, interpolation=cv2.INTER_AREA)
    template_copy = cv2.resize(template_copy, new_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite('template_copy.jpg', template_copy)
    rectangles, count_template_search=template_search(Magnitude_im, Magnitude_te)
    all_rectangles.extend(rectangles)
    image_rec_temp = drawing_rec(image_RGB_copy, rectangles)
    print("Найдено объектов после шаблонного поиска:", count_template_search)
    cv2.imwrite('Template_search.jpg', image_rec_temp)
    LBP_img=LBP_Matrix(template_copy)
    ghist1=LBP_Ghist_normalization(LBP_Ghist(LBP_img))
    cv2.imwrite('LBP_img.jpg', LBP_img)
    all_rectangles = merge_duplicate_rectangles(all_rectangles, threshold=0.3)
    rectangles_new=[]
    count_after_cheking = 0
    for rectangle in all_rectangles: 
        check, rectangle = cheking(image, rectangle, ghist1)
        if check is True:
            count_after_cheking += 1
            rectangles_new.append(rectangle)

    rectangles_new = merge_duplicate_rectangles(rectangles_new, threshold=0.3)
    count_after_cheking = len(rectangles_new)
    
    image_rec_check = drawing_rec(image_RGB, rectangles_new)
    print("Найдено объектов после проверки:", count_after_cheking)
    cv2.imwrite('Check.jpg', image_rec_check)

final_rectangles = merge_duplicate_rectangles(rectangles_new, threshold=0.3)
image_rec_final = drawing_rec(image_RGB, final_rectangles)
cv2.imwrite('Final_Check.jpg', image_rec_final)
print("Финальное количество уникальных объектов:", len(final_rectangles))