import os
import argparse
import cv2
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Загрузка CLIP на {DEVICE}...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

def get_embeddings_batch(crops):
    """Пакетное извлечение признаков CLIP для списка изображений-кропов."""
    imgs = []
    for c in crops:
        img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        imgs.append(clip_preprocess(img))

    imgs = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(imgs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

class YOLODetector:
    def __init__(self, weights, conf):
        self.model = YOLO(weights)
        self.conf = conf

    def predict(self, img):
        res = self.model.predict(img, conf=self.conf, verbose=False)[0]
        detections = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            score = float(b.conf[0])
            detections.append((x1, y1, x2, y2, score))
        return detections

class SahiDetector:
    def __init__(self, weights, conf):
        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights,
            confidence_threshold=conf,
        )

    def predict(self, img, slice_h, slice_w, overlap):
        result = get_sliced_prediction(
            img,
            self.model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=False
        )
        detections = []
        for p in result.object_prediction_list:
            x1, y1, x2, y2 = map(int, p.bbox.to_voc_bbox())
            score = float(p.score.value)
            detections.append((x1, y1, x2, y2, score))
        return detections

def crop_box(img, box):
    """Безопасная обрезка изображения по координатам рамки."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def filter_by_roi_similarity(img, detections, roi, threshold=0.75):
    """Фильтрация найденных объектов по косинусному сходству CLIP с эталоном."""
    roi_crop = crop_box(img, roi)
    if roi_crop is None:
        return detections

    roi_emb = get_embeddings_batch([roi_crop])[0].unsqueeze(0)
    crops, valid = [], []

    for d in detections:
        det_crop = crop_box(img, (d[0], d[1], d[2], d[3]))
        if det_crop is not None:
            crops.append(det_crop)
            valid.append(d)

    if not crops:
        return []

    det_embs = get_embeddings_batch(crops)
    results = []

    for i, d in enumerate(valid):
        sim = torch.nn.functional.cosine_similarity(roi_emb, det_embs[i].unsqueeze(0)).item()
        if sim > threshold:
            results.append(d)

    return results


def select_roi_ui(image):
    """Открывает окно для выделения эталонного объекта."""
    print("\n[UI] Выделите эталонный объект (ROI).")
    print("[UI] Нажмите SPACE или ENTER для подтверждения, 'c' для отмены.")
    
    r = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    x, y, w, h = r
    if w > 0 and h > 0:
        return (int(x), int(y), int(x + w), int(y + h))
    return None

def YOLOMain(image, model_path="YOLO/yolov8x.pt", use_sahi=False,
                  use_roi=True, conf=0.25, slice_h=128, slice_w=128,
                  overlap=0.2, sim_threshold=0.75):
    """Главная функция для импорта или консольного запуска."""

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f"Запуск детекции (SAHI: {use_sahi})...")
    if use_sahi:
        detector = SahiDetector(model_path, conf)
        detections = detector.predict(image_bgr, slice_h, slice_w, overlap)
    else:
        detector = YOLODetector(model_path, conf)
        detections = detector.predict(image_bgr)
    
    print(f"Первичная детекция нашла: {len(detections)} объектов")

    if use_roi:
        roi = select_roi_ui(image_bgr)

    if use_roi:
        print(f"Фильтрация CLIP (порог > {sim_threshold})...")
        detections = filter_by_roi_similarity(image, detections, roi, threshold=sim_threshold)

    print(f'\n=================================')
    print(f'ПРЕДСКАЗАННОЕ КОЛИЧЕСТВО: {len(detections)}')
    print(f'=================================\n')

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    if roi:
        plt.gca().add_patch(plt.Rectangle(
            (roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1], 
            edgecolor='red', facecolor='none', lw=3, label="Эталон (ROI)"
        ))
        plt.legend()
    plt.title("Исходное изображение")
    plt.axis('off')

    # Правый график: Результат
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    for (x1, y1, x2, y2, score) in detections:
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            edgecolor='lime', facecolor='none', lw=2
        ))
        plt.text(x1, y1-5, f"{score:.2f}", color='lime', fontsize=8, backgroundcolor='black')
        
    plt.title(f"Результат: {len(detections)} шт.")
    plt.axis('off')
    
    plt.show()

    # Возвращаем список только координат
    return [[d[0], d[1], d[2], d[3]] for d in detections]

if __name__ == "__main__":

    YOLOMain(cv2.cvtColor(cv2.imread("assets/5.jpg"), cv2.COLOR_BGR2RGB))