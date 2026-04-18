import cv2
import numpy as np

image = cv2.imread("Template/image/image.png", 0)

"""x1, y1 = 0, 0
x2, y2 = 60, 110"""

"""x1, y1 = 30, 43
x2, y2 = 70, 80"""

"""result = image[y1:y2, x1:x2]

cv2.imshow("result", result)
cv2.waitKey(0)

cv2.imwrite("result.png", result)"""


template = cv2.imread("Template/image/result.png", 0)
cv2.imshow("result", template)
cv2.waitKey(0)

w, h = template.shape[::-1]

result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
locations = np.where(result >= threshold)

count = 0

rectangles = []

for pt in zip(*locations[::-1]):
    rectangles.append([pt[0], pt[1], w, h])
    rectangles.append([pt[0], pt[1], w, h])

rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.5)

for (x, y, w, h) in rectangles:
    cv2.rectangle(image, (x,y), (x+w,y+h), 150, 2)
    count += 1

"""for pt in zip(*locations[::-1]):
    cv2.rectangle(image, pt, (pt[0]+w, pt[1]+h), 200, 2)
    count += 1
"""
print("Objects detected:", count)
cv2.imshow("result", image)
cv2.waitKey(0)


