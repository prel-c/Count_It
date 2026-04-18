import json
import os
import cv2
import numpy as np

def convert_yolo_to_fsc_json(img_dir, anno_dir, output_json):
    fsc_annotations = {}
    all_image_ids = []

    for img_name in os.listdir(img_dir):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_id = img_name
        all_image_ids.append(img_id)
        
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w = img.shape[:2]
        
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(anno_dir, txt_name)
        
        points = []
        bboxes = []
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    # YOLO: class x_center y_center width height (normalized)
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 5: continue
                    
                    # Денормализация
                    xc, yc = parts[1] * w, parts[2] * h
                    bw, bh = parts[3] * w, parts[4] * h
                    
                    points.append([float(xc), float(yc)])
                    
                    # (4 точки по часовой стрелке)
                    x1, y1 = xc - bw/2, yc - bh/2
                    x2, y2 = xc + bw/2, yc + bh/2
                    bboxes.append([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ])
        example_boxes = bboxes[:3] 
        
        fsc_annotations[img_id] = {
            'points': points,
            'box_examples_coordinates': example_boxes,
            'img_size': [h, w]
        }

    with open(output_json, 'w') as f:
        json.dump(fsc_annotations, f)
    
    return all_image_ids

def create_split_file(image_ids, output_split_json):
    np.random.shuffle(image_ids)
    n = len(image_ids)
    
    # Стандартное разделение 70/15/15
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    splits = {
        'train': image_ids[:train_end],
        'val': image_ids[train_end:val_end],
        'test': image_ids[val_end:]
    }
    
    with open(output_split_json, 'w') as f:
        json.dump(splits, f)

IMG_DIR = 'data/my/Pipes_340/img_new'
ANNO_DIR = 'data/my/Pipes_340/lab_new'
ids = convert_yolo_to_fsc_json(IMG_DIR, ANNO_DIR, 'annotation_my.json')
create_split_file(ids, 'Train_Test_my.json')