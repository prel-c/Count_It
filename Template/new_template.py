import cv2

image = cv2.imread("ResNet/data/images_384_VarV2/1.jpg")
clone = image.copy()

drawing = False
x0, y0 = -1, -1
x1, y1 = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global x0, y0, x1, y1, drawing, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            cv2.rectangle(img, (x0, y0), (x, y), (0,255,0), 2)
            cv2.imshow("image", img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = x, y
        cv2.rectangle(image, (x0,y0), (x1,y1), (0,255,0), 2)
        cv2.imshow("image", image)

cv2.imshow("image", image)
cv2.setMouseCallback("image", draw_rectangle)

cv2.waitKey(0)

template = clone[min(y0,y1):max(y0,y1), min(x0,x1):max(x0,x1)]
print(min(y0,y1), max(y0,y1), min(x0,x1), max(x0,x1))

cv2.imshow("template", template)
cv2.waitKey(0)
cv2.imwrite("Template/image/result.png", template)