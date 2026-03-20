import numpy as np
import cv2
import math

photo=cv2.imread("photo2.webp")
photo=cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
photo = photo.astype(np.float32)


Direction=np.zeros((len(photo), len(photo[0])), dtype=np.float32)
Magnitude=np.zeros((len(photo), len(photo[0])), dtype=np.float32)


for i in range (1, len(photo)-1, 1):
    for j in range (1, len(photo[0])-1, 1):
        Y=photo[i+1, j]-photo[i-1, j]
        X=photo[i, j+1]-photo[i, j-1]
        Magnitude[i, j]=math.sqrt(X**2+Y**2)
        if X!=0:
            Direction[i, j]=math.atan2(Y, X)%math.pi
        else:
            Direction[i, j]=math.pi/2

Cells=np.empty((len(photo)//8, len(photo[0])//8), dtype=object)

Bins=[[0, 20/180*math.pi, 0], 
      [20/180*math.pi, 40/180*math.pi, 1], 
      [40/180*math.pi, 60/180*math.pi, 2], 
      [60/180*math.pi, 80/180*math.pi, 3], 
      [80/180*math.pi, 100/180*math.pi, 4],
      [100/180*math.pi, 120/180*math.pi, 5], 
      [120/180*math.pi, 140/180*math.pi, 6], 
      [140/180*math.pi, 160/180*math.pi, 7], 
      [160/180*math.pi, math.pi, 8]]



Cells = np.zeros((len(photo) // 8, len(photo[0]) // 8, 9), dtype=np.float32)

for i in range(len(photo) // 8):
    for j in range(len(photo[0]) // 8):
        cell = np.zeros(9)
        for i1 in range(i * 8, (i + 1) * 8):
            for j1 in range(j * 8, (j + 1) * 8):
                for s, e, n in Bins:
                    if s <= Direction[i1, j1] < e:
                        cell[n] += Magnitude[i1, j1]
                        break
        Cells[i, j] = cell



e = 1e-6      
blocks = []
for i in range(len(photo)//8-1):
    for j in range(len(photo[0])//8-1):
        Megavector=np.zeros(36, dtype=np.float32)
        Megavector[0:9]=Cells[i, j]
        Megavector[9:18]=Cells[i + 1, j]
        Megavector[18:27]=Cells[i, j + 1]
        Megavector[27:36]=Cells[i + 1, j + 1]
        l2 = math.sqrt(np.sum(Megavector**2) + e**2)
        Megavector = Megavector/l2
        Megavector = np.clip(Megavector, 0, 0.2)
        l2 = math.sqrt(np.sum(Megavector**2) + e**2)
        Megavector = Megavector / l2
        blocks.append(Megavector)
        



import matplotlib.pyplot as plt


cell_size = 8
bin_width = 180 / 9
angles = np.arange(0, 180, bin_width) + bin_width / 2
angles_rad = np.deg2rad(angles)


hog_image = np.zeros_like(photo)
rows, cols = Cells.shape[:2]

for i in range(rows):
    for j in range(cols):
        cx, cy = (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2)
        for bin_idx in range(9):
            magnitude = Cells[i, j, bin_idx]
            if magnitude == 0: continue
            length = (magnitude / (np.max(Cells) + 1e-6)) * cell_size * 0.8
            dx = length * math.cos(angles_rad[bin_idx])
            dy = length * math.sin(angles_rad[bin_idx])
            p1 = (int(cx - dx), int(cy - dy))
            p2 = (int(cx + dx), int(cy + dy))
            brightness = min(255, int(255 * (magnitude /    20)))
            cv2.line(hog_image, p1, p2, brightness, 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Оригинал (Gray)")
plt.imshow(photo, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Визуализация HOG")
plt.imshow(hog_image, cmap='inferno')
plt.savefig('Ghist_obj.png')



Grad_X = np.zeros_like(photo)
Grad_Y = np.zeros_like(photo)

for i in range(1, len(photo)-1):
    for j in range(1, len(photo[0])-1):
        Y = photo[i+1, j] - photo[i-1, j]
        X = photo[i, j+1] - photo[i, j-1]
        
        Grad_X[i, j] = X
        Grad_Y[i, j] = Y
        Magnitude[i, j] = math.sqrt(X**2 + Y**2)

def normalize_img(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

cv2.imwrite("Magnitude.png", normalize_img(Magnitude))
cv2.imwrite("x.png", normalize_img(Grad_X))
cv2.imwrite("y.png", normalize_img(Grad_Y))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.title("Градиент X"); plt.imshow(Grad_X, cmap='gray')
plt.subplot(1, 3, 2); plt.title("Градиент Y"); plt.imshow(Grad_Y, cmap='gray')
plt.subplot(1, 3, 3); plt.title("Магнитуда"); plt.imshow(Magnitude, cmap='magma')
plt.savefig('All.png')