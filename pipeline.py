import cv2
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# 1. Инициализация

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGE_SIZE = 768

# CLIP

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

# SAM

sam_checkpoint = "sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to("cuda")

predictor = SamPredictor(sam)

mask_generator = SamAutomaticMaskGenerator(
sam,
points_per_side=12, #16
pred_iou_thresh=0.8,
stability_score_thresh=0.9,
# box_nms_thresh=0.3,
min_mask_region_area=500 #500
)


# 2. Utils

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]
    scale = 1.0

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, new_h))

    return image, scale

def crop_object(image, mask, target_size=224):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Оставляем немного фона для контекста
    pad = int(max(x2 - x1, y2 - y1) * 0.15)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)

    x2 = min(image.shape[1], x2 + pad)
    y2 = min(image.shape[0], y2 + pad)

    crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    crop = crop.copy()
    crop[~mask_crop] = (crop[~mask_crop] * 0.5).astype(np.uint8)

    h, w = crop.shape[:2]

    # Делаем квадрат через padding
    size = max(h, w)
    square = np.zeros((size, size, 3), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[y_offset:y_offset + h, x_offset:x_offset + w] = crop

    # --- теперь resize БЕЗ искажений ---
    square = cv2.resize(square, (target_size, target_size))

    return square


# 3. Feature extraction (CLIP)

def extract_feature(image_crop):
    # numpy (H, W, C) → PIL
    image_pil = Image.fromarray(image_crop)

    image = clip_preprocess(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = clip_model.encode_image(image)

    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze()

def extract_features(crop):

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    soft_color = cv2.addWeighted(crop, 0.7, gray_rgb, 0.3, 0)

    variants = []

    variants.append(soft_color)
    variants.append(cv2.flip(soft_color, 1))

    feats = []

    for img in variants:
        f = extract_feature(img)
        feats.append(f)

    return np.stack(feats)

def match_features(ref_feats, obj_feats):
    dists = []

    for rf in ref_feats:
        for of in obj_feats:
            dists.append(1 - np.dot(rf, of))

    return min(dists)

# 4. Эталон
def get_reference(image, point_xy):
    image_resized, scale = resize_image(image)
    point_scaled = (int(point_xy[0] * scale), int(point_xy[1] * scale))

    predictor.set_image(image_resized)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([point_scaled]),
        point_labels=np.array([1]),
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]
    # print(type(best_mask))

    crop = crop_object(image_resized, best_mask)
    if crop is None:
        raise ValueError("Не удалось выделить объект")

    features = extract_features(crop)
    # print('признаки эталона:', features[0][:20])

    return features, image_resized, best_mask, point_scaled


# 5. Поиск похожих объектов

def cosine_distance(a, b):
    return 1 - np.dot(a, b)


def compute_threshold(distances: list, direction: str = 'left', k_sigma: float = 2.0,
                      peak_threshold: float = 0.25, plot: bool = False) -> float:
    """
    Определяет порог на основе пика гистограммы и подогнанного нормального распределения.

    Параметры:
        distances: list of float — исходные данные.
        direction: 'left' или 'right' — отступать от правого конца влево (left) или вправо (right).
        k_sigma: число сигм для определения правого конца (mean + k_sigma * std).
        peak_threshold: доля от максимальной частоты, чтобы отобрать бины вокруг пика.
        plot: если True, строит график гистограммы и подогнанной кривой.

    Возвращает:
        Пороговое значение (float).
    """
    distances = np.asarray(distances)
    if len(distances) == 0:
        raise ValueError("Список distances пуст")

    # 1. Построение гистограммы
    n_bins = int(np.sqrt(len(distances))) + 1
    hist, bin_edges = np.histogram(distances, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Поиск пика гистограммы
    peak_idx = np.argmax(hist)
    peak_center = bin_centers[peak_idx]
    max_freq = hist[peak_idx]

    # 3. Отбор данных вокруг пика (бины с частотой >= peak_threshold * max_freq)

    selected_bins = hist >= (peak_threshold * max_freq)
    if not np.any(selected_bins):
        # если ничего не выбрано, оставляем только пиковый бин
        selected_bins[peak_idx] = True

    # Левый и правый края отобранных бинов
    first_selected_idx = np.where(selected_bins)[0][0]
    last_selected_idx = np.where(selected_bins)[0][-1]
    left_bound = bin_edges[first_selected_idx]
    right_bound = bin_edges[last_selected_idx + 1]

    # Выбираем исходные точки, попадающие в этот интервал
    data_subset = distances[(distances >= left_bound) & (distances <= right_bound)]

    # Защита: если данных недостаточно, возвращаем медиану
    if len(data_subset) < 2:
        return float(np.median(distances))

    # 4. Подгонка нормального распределения (MLE)
    mean = np.mean(data_subset)
    std = np.std(data_subset, ddof=1)   # несмещённая оценка

    # 5. Правый конец горки (mean + k_sigma * std)
    right_end = mean + k_sigma * std

    # 6. Отступ на две сигмы в заданном направлении
    if direction == 'left':
        threshold = right_end - 2 * std
    else:   # direction == 'right'
        threshold = right_end + 2 * std

    return float(threshold)

"""def compute_threshold(distances):
    d = np.sort(distances)
    return np.median(d)+0.05

    if len(d) < 2:
        return 0.25

    d = d[2:]

    gaps = np.diff(d)

    left_points = d[:-1]

    # Эмпирически
    weights = np.exp(-0.5 * left_points)

    weighted_gaps = gaps * weights

    best = np.argmax(weighted_gaps)

    # t = (d[best] + d[best + 1]) / 2
    t = d[best]

    # ограничиваем threshold
    #t = max(t, 0.08)
    t = min(t, 0.25)

    return t"""

def find_similar_objects(image, ref_feature):
    image_resized, _ = resize_image(image)

    masks = mask_generator.generate(image_resized)

    # Визуализация всех масок SAM
    plt.figure(figsize=(12, 12))
    plt.imshow(image_resized)

    for ann in masks:
        mask = ann["segmentation"]

        # случайная прозрачная маска
        color = np.random.rand(3)

        overlay = np.zeros((*mask.shape, 4))
        overlay[..., :3] = color
        overlay[..., 3] = mask * 0.35

        plt.imshow(overlay)

    plt.title(f"Все маски SAM ({len(masks)})")
    plt.axis("off")
    plt.show()

    # print(f"Всего масок: {len(masks)}")

    #print(f"Всего масок SAM: {len(masks)}")

    candidates = []

    for ann in masks:
        mask = ann["segmentation"]

        crop = crop_object(image_resized, mask)
        if crop is None:
            continue


        # 1. извлекаем признаки объекта
        obj_feats = extract_features(crop)
        # print('признаки масок:', obj_feats[0][:20])

        # 2. сравниваем с эталоном
        dist = match_features(ref_feature, obj_feats)

        candidates.append((mask, dist))

    if len(candidates) == 0:
        return [], image_resized, []

    dists = [d for _, d in candidates]
    threshold = compute_threshold(dists)

    # print(sorted(dists)[:20])
    # print("Auto threshold:", threshold)

    results = [(m, d) for (m, d) in candidates if d < threshold]


    """# Визуализация масок-кандидатов
    for mask, dist in results:
        overlay = np.zeros((*mask.shape, 4))
        overlay[..., :3] = np.random.rand(3)
        overlay[..., 3] = mask * 0.45

        plt.imshow(overlay)"""

    plt.title(f"Отобранные объекты ({len(results)})")
    plt.axis("off")
    plt.show()

    if len(results) == 0:
        best = min(candidates, key=lambda x: x[1])
        results = [best]

    return results, image_resized, dists, threshold


# 6. Визуализация

"""def show_results(image, masks_with_dist, ref_point=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for mask, dist in masks_with_dist:
        plt.imshow(mask, alpha=0.4)

    if ref_point is not None:
        plt.scatter(ref_point[0], ref_point[1], c="red", s=100)

    plt.title(f"Найдено: {len(masks_with_dist)}")
    plt.axis("off")
    plt.show()"""


def show_results(image, masks_with_dist, ref_point=None):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)

    for mask, dist in masks_with_dist:

        # случайный цвет маски
        color = np.random.rand(3)

        # создаём RGBA overlay
        overlay = np.zeros((*mask.shape, 4))

        # RGB
        overlay[..., :3] = color

        # прозрачность только внутри маски
        overlay[..., 3] = mask.astype(float) * 0.45

        plt.imshow(overlay)

    # точка эталона
    if ref_point is not None:
        plt.scatter(
            ref_point[0],
            ref_point[1],
            c="red",
            s=120,
            edgecolors="white",
            linewidths=2
        )

    plt.title(f"Найдено объектов: {len(masks_with_dist)}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def get_click_point(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Кликните по объекту")

    point = []

    def onclick(event):
        if event.inaxes:
            point.append((int(event.xdata), int(event.ydata)))
            plt.close()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if not point:
        raise ValueError("Точка не выбрана")

    return point[0]