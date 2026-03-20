import cv2
import numpy as np
import matplotlib.pyplot as plt


# Радиус - 1 пиксель. Длина - 4 пикселя.

def LBP_1_4(i, j, photo):
    st=[]
    lbp=0
    px = [(i-1, j),
          (i, j+1),
          (i+1, j),
          (i, j-1)]
    
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


# Радиус - 1 пиксель. Длина - 8 пикселей.

# Самый стабильный -------------------------------------------------------
def LBP_1_8(i, j, photo):
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
#-------------------------------------------------------------------------

# Радиус - 2 пикселя. Длина - 12 пикселей.

def LBP_2_12(i, j, photo):
    st=[]
    lbp=0
    px = [(i-2, j),
          (i-2, j+1),
          (i-1, j+2),
          (i, j+2),
          (i+1, j+2),
          (i+2, j+1),
          (i+2, j),
          (i+2, j-1),
          (i+1, j-2),
          (i, j-2),
          (i-1, j-2),
          (i-2, j-1),]
    if 0<i<len(photo)-2 and 0<j<len(photo[0])-2:
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

# Чтение изображения
photo=cv2.imread("photo2.webp")
photo=cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

# Создание LBP матрицы
lbp_img=np.full((len(photo), len(photo[0])), np.nan)

for i in range (1, len(photo)):
    for j in range (1, len(photo[0])):
        lbp_img[i, j]=LBP_1_8(i, j, photo)

cv2.imwrite('lbp_img_obj.jpg', lbp_img)

# Гистограмма количества равных LBP

Ghist_obj=np.zeros(256)
for i in range(len(Ghist_obj)):
    Ghist_obj[i]=np.count_nonzero(lbp_img == i)

plt.figure(figsize=(12, 6))
plt.bar(range(256), Ghist_obj, width=1.0)
plt.xlabel('Значение LBP')
plt.ylabel('Частота')
plt.title('Гистограмма LBP')
plt.grid(True, alpha=0.3)
plt.savefig('Ghist_obj.png')
