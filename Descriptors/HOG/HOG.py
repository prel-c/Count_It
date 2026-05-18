import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

photo=cv2.imread("photo2.webp")
photo=cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
photo = photo.astype(np.float32)


Direction=np.zeros((len(photo), len(photo[0])), dtype=np.float32)
Magnitude=np.zeros((len(photo), len(photo[0])), dtype=np.float32)
Y=np.zeros((len(photo), len(photo[0])), dtype=np.float32)
X=np.zeros((len(photo), len(photo[0])), dtype=np.float32)

for i in range (1, len(photo)-1, 1):
    for j in range (1, len(photo[0])-1, 1):
        Y[i, j]=photo[i+1, j]-photo[i-1, j]
        X[i, j]=photo[i, j+1]-photo[i, j-1]
        Magnitude[i, j]=math.sqrt(X[i, j]**2+Y[i, j]**2)
        if X[i, j]!=0:
            Direction[i, j]=math.atan2(Y[i, j], X[i, j])%math.pi
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

with open('blocks.txt', 'w', encoding='utf-8') as f:
    for row in blocks:
        f.write(' '.join(map(str, row)) +"|"+'\n')


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Градиент X")
plt.imshow(X/sum(X**2+e**2), cmap='grey')

plt.subplot(1, 3, 2)
plt.title("Градиент Y")
plt.imshow(Y/sum(Y**2+e**2), cmap='grey')

plt.subplot(1, 3, 3)
plt.title("Магнитуда")
plt.imshow((Magnitude/sum(Magnitude**2+e**2))**0.5, cmap='grey')

plt.savefig('All.png')



