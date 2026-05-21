import cv2
import torch
import clip

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load(
    "ViT-B/32",
    device=device
)


def get_embeddings_batch(crops):
    imgs = []

    for c in crops:
        img = Image.fromarray(
            cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        )

        imgs.append(
            clip_preprocess(img)
        )

    imgs = torch.stack(imgs).to(device)

    with torch.no_grad():
        emb = clip_model.encode_image(imgs)

    emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb


class YOLODetector:

    def __init__(self, weights, conf):
        self.model = YOLO(weights)
        self.conf = conf

    def predict(self, img):

        res = self.model.predict(
            img,
            conf=self.conf,
            verbose=False
        )[0]

        detections = []

        for b in res.boxes:

            x1, y1, x2, y2 = map(
                int,
                b.xyxy[0]
            )

            score = float(b.conf[0])

            detections.append(
                (x1, y1, x2, y2, score)
            )

        return detections


class SahiDetector:

    def __init__(self, weights, conf):

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights,
            confidence_threshold=conf,
        )

    def predict(
        self,
        img,
        slice_h,
        slice_w,
        overlap
    ):

        result = get_sliced_prediction(
            img,
            self.model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
        )

        detections = []

        for p in result.object_prediction_list:

            x1, y1, x2, y2 = map(
                int,
                p.bbox.to_voc_bbox()
            )

            score = float(p.score.value)

            detections.append(
                (x1, y1, x2, y2, score)
            )

        return detections


def crop(img, box):

    h, w = img.shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))

    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def select_roi(
    img,
    detections,
    roi,
    threshold=0.75
):

    roi_crop = crop(img, roi)

    if roi_crop is None:
        return detections

    roi_emb = get_embeddings_batch(
        [roi_crop]
    )[0].unsqueeze(0)

    crops = []
    valid = []

    for d in detections:

        x1, y1, x2, y2, score = d

        det_crop = crop(
            img,
            (x1, y1, x2, y2)
        )

        if det_crop is None:
            continue

        crops.append(det_crop)
        valid.append(d)

    if len(crops) == 0:
        return []

    det_embs = get_embeddings_batch(crops)

    results = []

    for i, d in enumerate(valid):

        sim = torch.nn.functional.cosine_similarity(
            roi_emb,
            det_embs[i].unsqueeze(0)
        ).item()

        if sim > threshold:
            results.append(d)

    return results


def draw(img, detections):

    for x1, y1, x2, y2, score in detections:

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return img


def yolo(
    image_path,
    model_path="yolov8x.pt",
    conf=0.25,
    use_sahi=False,
    slice_h=128,
    slice_w=128,
    overlap=0.2,
    use_roi=False,
    sim_threshold=0.75,
    save_path="result.jpg",
    return_image=False,
):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Image not found: {image_path}"
        )

    if use_sahi:

        detector = SahiDetector(
            model_path,
            conf
        )

        detections = detector.predict(
            img,
            slice_h,
            slice_w,
            overlap
        )

    else:

        detector = YOLODetector(
            model_path,
            conf
        )

        detections = detector.predict(img)

    roi = None

    if use_roi:

        r = cv2.selectROI(
            "Select ROI",
            img,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyAllWindows()

        x, y, w, h = r

        if w > 0 and h > 0:

            roi = (
                x,
                y,
                x + w,
                y + h
            )

        else:
            print("ROI cancelled")

    if roi is not None:

        detections = select_roi(
            img,
            detections,
            roi,
            threshold=sim_threshold,
        )

    result = draw(
        img.copy(),
        detections
    )

    cv2.imwrite(
        save_path,
        result
    )

    output = {
        "detections": detections,
        "count": len(detections),
        "result_image": result,
        "save_path": save_path,
    }

    print(
        f"Done. Found: {len(detections)} objects"
    )

    if return_image:
        return output

    return len(detections)


if __name__ == "__main__":

    yolo(
        image_path="test.jpg",
        model_path="yolov8x.pt",
        conf=0.25,
        use_sahi=True,
        use_roi=True,
        sim_threshold=0.75,
        save_path="result.jpg",
    )