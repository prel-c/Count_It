import cv2

img=cv2.imread("image.png")
img = cv2.GaussianBlur(img, (5, 5), 1)
cv2.imwrite("img.jpg", img)
