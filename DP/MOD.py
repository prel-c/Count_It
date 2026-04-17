import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
____________________________________________________

Функции реализации метода LBP
____________________________________________________

"""
# Поиск LBP R=1 P=8
def Uniform_LBP_1_8(i, j, photo, uniform_pattern, count_of_uniform_pattern):
    st=""
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
                st+="1"
            else:
                st+="0"
    
    
    for i in range(count_of_uniform_pattern):
        if str(uniform_pattern[i]) == str(st):
            return i
    return count_of_uniform_pattern
    


# Поиск LBP R=2 P=16
def LBP_2_16(i, j, photo):
    st=[]
    lbp=0
    px = [(i-2, j),
          (i-2, j+1),
          (i-2, j+2),
          (i-1, j+2),
          (i, j+2),
          (i+1, j+2),
          (i+2, j+2),
          (i+2, j+1),
          (i+2, j),
          (i+2, j-1),
          (i+2, j-2),
          (i+1, j-2),
          (i, j-2),
          (i-1, j-2),
          (i-2, j-2),
          (i-2, j-1),]
    if 0<i<len(photo)-2 and 0<j<len(photo[0])-2:
        for i1, j1 in px:
            if photo[i1, j1]>photo[i, j]:
                st.append(1)
            else:
                st.append(0)
        for i in range (len(st)):
            lbp+=st[i]*2**(len(st)-1-i)
        return lbp
    else:
        return 0

# Создаём список полезных LBP
def uniform_patterns(n):

    def binary_strings(length):
        total = 1 << length  
        result = [''] * total
        for i in range(total):
            result[i] = format(i, f'0{length}b')
        return result
    
    def checking_for_uniform_patterns(LBP):
        k = 0
        for i in range(len(LBP)):
            if LBP[i] != LBP[(i+1) % len(LBP)]:
                k += 1
                if k > 2:
                    return False
        return True

    result=[]
    bin=binary_strings(n)
    for i in bin:
        if checking_for_uniform_patterns(i) is True:
            result.append(i)
    return result


# Состевление нормализованной гистограммы Uniform LBP для объекта
def LBP_Ghist(photo, count_of_uniform_pattern, uniform_pattern):
    ghistogramm=np.zeros(count_of_uniform_pattern+1)
    for i in range (1, len(photo)):
        for j in range (1, len(photo[0])):
            ghistogramm[Uniform_LBP_1_8(i, j, photo, uniform_pattern, count_of_uniform_pattern)]+=1
    return ghistogramm/np.sum(ghistogramm)
    


# Сраванение гистограмм
def chi_square(ghist1, ghist2, epsilon=1e-10):
    return np.sum((ghist1 - ghist2) ** 2 / (ghist1 + ghist2 + epsilon))

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


def cheking(image, rectangle, ghist1, count_of_uniform_pattern, uniform_pattern):
    x=rectangle[0]
    y=rectangle[1]
    w=rectangle[2]
    h=rectangle[3]
    img=image[y:y+h, x:x+w]
    ghist2=LBP_Ghist(img, count_of_uniform_pattern, uniform_pattern)
    plt.figure(figsize=(12, 6))
    plt.bar(range(57), ghist2[1:-1], width=1.0)
    plt.xlabel('Значение LBP')
    plt.ylabel('Частота')
    plt.title('Гистограмма LBP')
    plt.grid(True, alpha=0.3)
    plt.savefig('Ghist_obj.png')
    if chi_square(ghist1[1:-1], ghist2[1:-1], epsilon=1e-10) < 0.15:
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

# Ввод изображения и обьекта
image_RGB = cv2.imread("image.jpg")
template_RGB = cv2.imread("image1.jpg")

image_RGB_copy = np.copy(image_RGB)

image=cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
image_copy=np.copy(image)
template=cv2.cvtColor(template_RGB, cv2.COLOR_BGR2GRAY)
template_copy=np.copy(template)


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

# Поиск гистограммы объекта

uniform_pattern=uniform_patterns(8)
count_of_uniform_pattern=len(uniform_pattern)
ghist1=LBP_Ghist(template_copy, count_of_uniform_pattern, uniform_pattern)

plt.figure(figsize=(12, 6))
plt.bar(range(57), ghist1[1:-1], width=1.0)
plt.xlabel('Значение LBP')
plt.ylabel('Частота')
plt.title('Гистограмма LBP')
plt.grid(True, alpha=0.3)
plt.savefig('Ghist1_obj.png')

# Проверка шаблонного поиска
rectangles_new=[]
count_after_cheking = 0
for rectangle in rectangles:
    check, rectangle = cheking(image, rectangle, ghist1, count_of_uniform_pattern, uniform_pattern)
    if check is True:
        count_after_cheking += 1
        rectangles_new.append(rectangle)
image_rec_check = drawing_rec(image_RGB, rectangles_new)

print("Найдено объектов после проверки:", count_after_cheking)
cv2.imwrite('Check.jpg', image_rec_check)