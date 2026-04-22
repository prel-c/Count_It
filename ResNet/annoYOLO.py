import os
import cv2
import matplotlib.pyplot as plt
import os


def visualize_yolo_annotations(image_path, txt_path, save_path=None):
    """
    Рисует боксы из YOLO-аннотации на изображении.
    """
    image = cv2.imread(image_path)

    img_h, img_w, _ = image.shape

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        class_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])

        x_center = x_center_norm * img_w
        y_center = y_center_norm * img_h
        w_pixel = w_norm * img_w
        h_pixel = h_norm * img_h

        x1 = int(x_center - (w_pixel / 2))
        y1 = int(y_center - (h_pixel / 2))
        x2 = int(x_center + (w_pixel / 2))
        y2 = int(y_center + (h_pixel / 2))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.circle(image, (int(x_center), int(y_center)), 2, (0, 0, 255), -1)

    if save_path:
        cv2.imwrite(save_path, image)
        print(f"Результат сохранен в: {save_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Найдено объектов: {len(lines)}")
    plt.show()

data_folder = "ResNet/data" 
img_file = os.path.join(data_folder, "images_384_VarV2", "350.jpg")
txt_file = os.path.join(data_folder, "yolo_labels", "350.txt")

visualize_yolo_annotations(
    image_path=img_file, 
    txt_path=txt_file, 
    # save_path="test_visualization.jpg" 
)