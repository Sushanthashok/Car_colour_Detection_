# detect_pipeline.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

from color_detect import is_blue_car_hsv  # your standalone HSV detector

# Load detection model
DETECT_PATH = "yolov8n.pt"
detect_model = YOLO(DETECT_PATH)

# Optional classifier path - set to None or ensure file exists
CLS_MODEL_PATH = "runs/classify/train2/weights/best.pt"
cls_model = None
if CLS_MODEL_PATH and os.path.exists(CLS_MODEL_PATH):
    try:
        cls_model = YOLO(CLS_MODEL_PATH)
    except Exception:
        cls_model = None

CAR_LABELS = {"car", "truck", "bus"}
PERSON_LABEL = "person"


def _safe_get_names(model):
    try:
        return model.model.names
    except Exception:
        try:
            return model.names
        except Exception:
            return {}


def _probs_to_numpy(probs):
    """
    Safely convert various ultralytics 'probs' objects to a 1D numpy array or None.
    Handles:
     - torch.Tensor
     - np.ndarray
     - ultralytics.Probs (has .numpy() and .cpu())
     - lists/iterables
    """
    if probs is None:
        return None

    # torch tensor
    try:
        import torch

        if isinstance(probs, torch.Tensor):
            return probs.cpu().numpy()
    except Exception:
        pass

    # ultralytics Probs-like object: has .numpy()
    try:
        if hasattr(probs, "numpy") and callable(getattr(probs, "numpy")):
            arr = probs.numpy()
            return np.asarray(arr)
    except Exception:
        pass

    # object with cpu().numpy()
    try:
        if hasattr(probs, "cpu") and callable(getattr(probs, "cpu")):
            cpu_obj = probs.cpu()
            if hasattr(cpu_obj, "numpy") and callable(getattr(cpu_obj, "numpy")):
                return np.asarray(cpu_obj.numpy())
    except Exception:
        pass

    # numpy array
    if isinstance(probs, np.ndarray):
        return probs

    # fallback: try to coerce to array (list, tuple, iterable)
    try:
        return np.asarray(list(probs))
    except Exception:
        pass

    return None


def classify_crop_with_model(crop_bgr):
    """
    Classify crop using trained classifier (if available).
    Returns (label:str, conf:float) or ("unknown", 0.0).
    """
    if cls_model is None:
        return "unknown", 0.0
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0

    # convert to RGB as classifier expects RGB
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        crop_rgb = crop_bgr[..., ::-1]

    try:
        results = cls_model.predict(source=crop_rgb, imgsz=224, verbose=False)
    except Exception:
        return "unknown", 0.0

    # get first result gracefully
    try:
        res = results[0]
    except Exception:
        res = results

    # try res.probs first
    probs_candidate = None
    if hasattr(res, "probs"):
        try:
            probs_candidate = res.probs
        except Exception:
            probs_candidate = None

    # convert to numpy
    probs_np = _probs_to_numpy(probs_candidate)

    # fallback: try res.data
    if probs_np is None or np.asarray(probs_np).size == 0:
        try:
            if hasattr(res, "data"):
                probs_np = _probs_to_numpy(res.data)
        except Exception:
            probs_np = None

    # still nothing -> unknown
    if probs_np is None or np.asarray(probs_np).size == 0:
        return "unknown", 0.0

    # compute best index & confidence
    try:
        idx = int(np.argmax(probs_np))
        conf = float(np.max(probs_np))
    except Exception:
        return "unknown", 0.0

    # get names safely
    names = cls_model.names
    if isinstance(names, dict):
        label = names.get(idx, str(idx))
    else:
        label = names[idx] if idx < len(names) else str(idx)

    return label, conf


def process_frame(bgr, blue_threshold=0.08, conf_thresh=0.35):
    """
    Run YOLO detection on BGR frame, perform HSV color test on each detected car crop.
    Returns: annotated_bgr, total_cars, blue_car_count, other_car_count, people_count
    """
    if bgr is None:
        raise ValueError("frame is None")

    results = detect_model.predict(source=bgr, conf=conf_thresh, verbose=False)
    r = results[0] if isinstance(results, (list, tuple)) else results

    names_map = _safe_get_names(detect_model)
    canvas = bgr.copy()
    h, w = canvas.shape[:2]

    total_cars = blue_cars = other_cars = people_count = 0

    for box in r.boxes:
        # class id robust extraction
        try:
            cls_id = int(box.cls.item())
        except Exception:
            try:
                cls_id = int(box.cls.cpu().numpy()[0])
            except Exception:
                cls_id = int(box.cls[0])

        cls_name = names_map.get(cls_id, str(cls_id))

        # coords robust extraction
        try:
            coords = box.xyxy[0].cpu().numpy()
        except Exception:
            try:
                coords = box.xyxy[0].numpy()
            except Exception:
                coords = box.xyxy[0]
        x1, y1, x2, y2 = [int(c) for c in coords]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if cls_name == PERSON_LABEL:
            people_count += 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)
            continue

        if cls_name in CAR_LABELS:
            total_cars += 1
            crop = canvas[y1:y2, x1:x2]

            is_blue, blue_frac = is_blue_car_hsv(crop, blue_thresh=blue_threshold)
            cls_label, cls_conf = classify_crop_with_model(crop) if cls_model else ("", 0.0)

            color = (0, 0, 255) if is_blue else (255, 0, 0)
            label_text = f"{'BLUE' if is_blue else 'OTHER'} {blue_frac:.2f}"
            if cls_label and cls_label != "unknown":
                label_text = f"{cls_label} {cls_conf:.2f} | {label_text}"

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(canvas, label_text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            if is_blue:
                blue_cars += 1
            else:
                other_cars += 1

    return canvas, total_cars, blue_cars, other_cars, people_count


