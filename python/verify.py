# LOAD LIBRARIES
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from inspect_onnx import (
    preprocess,
    infer,
    postprocess,
    iou
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

model_path = "models/visdrone_yolo11n_640.onnx"
image_path = "image/0000242_02400_d_0000009.jpg"


# --------------------------------------------------
# ONNX PIPELINE
# --------------------------------------------------

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

image = cv2.imread(image_path)

if image is None:
    raise ValueError(
        f"Could not load image: {image_path}"
    )

orig_h, orig_w = image.shape[:2]

tensor, scale, pad_x, pad_y = preprocess(
    image,
    input_size=640
)

raw_output = infer(
    session,
    tensor
)

detections = postprocess(
    raw_output,
    scale,
    pad_x,
    pad_y,
    orig_w,
    orig_h,
    conf_threshold=0.25,
    iou_threshold=0.7
)


# --------------------------------------------------
# ULTRALYTICS REFERENCE
# --------------------------------------------------

ref_model = YOLO(
    model_path,
    task="detect"
)

ref = ref_model.predict(
    source=image_path,
    imgsz=640,
    conf=0.25,
    iou=0.7,
    max_det=1000,
    device="cpu",
    verbose=False
)[0]

ref_boxes = ref.boxes.xyxy.cpu().numpy()
ref_classes = ref.boxes.cls.cpu().numpy().astype(int)
ref_scores = ref.boxes.conf.cpu().numpy()


# --------------------------------------------------
# CONVERT DETECTIONS TO ARRAYS
# --------------------------------------------------

boxes = np.array(
    [det["box"] for det in detections],
    dtype=np.float32
)

classes = np.array(
    [det["class_id"] for det in detections],
    dtype=np.int32
)

scores = np.array(
    [det["confidence"] for det in detections],
    dtype=np.float32
)


# --------------------------------------------------
# MATCH AGAINST ULTRALYTICS
# --------------------------------------------------

def compare_detections(
    ref_boxes,
    ref_classes,
    ref_scores,
    boxes,
    classes,
    scores,
    match_iou=0.5
):
    used = set()

    matched_ious = []
    confidence_diffs = []

    matched = 0

    for ref_idx in range(len(ref_boxes)):

        best_iou = 0.0
        best_idx = -1

        for idx in range(len(boxes)):

            if idx in used:
                continue

            if ref_classes[ref_idx] != classes[idx]:
                continue

            overlap = iou(
                ref_boxes[ref_idx],
                boxes[idx]
            )

            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx

        if best_idx != -1 and best_iou >= match_iou:
            used.add(best_idx)

            matched += 1
            matched_ious.append(best_iou)

            confidence_diffs.append(
                abs(
                    float(ref_scores[ref_idx])
                    - float(scores[best_idx])
                )
            )

    return matched, matched_ious, confidence_diffs


# --------------------------------------------------
# RUN COMPARISON
# --------------------------------------------------

matched, matched_ious, confidence_diffs = compare_detections(
    ref_boxes,
    ref_classes,
    ref_scores,
    boxes,
    classes,
    scores
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("Ultralytics count:", len(ref_boxes))
print("Count:", len(boxes))
print("Matched:", matched)

if matched_ious:
    print(
        "Lowest matched IoU:",
        min(matched_ious)
    )

    print(
        "Average matched IoU:",
        np.mean(matched_ious)
    )

if confidence_diffs:
    print(
        "Largest confidence difference:",
        max(confidence_diffs)
    )

# TEST
shifted_boxes = boxes.copy()
shifted_boxes[:, [0, 2]] += 5.0

matched_test, ious_test, confidence_diffs_test = compare_detections(
    ref_boxes,
    ref_classes,
    ref_scores,
    shifted_boxes,
    classes,
    scores
)

print("Shifted matched:", matched_test)
print("Shifted lowest IoU:", min(ious_test))
print("Shifted average IoU:", np.mean(ious_test))

### Step 4.6 — Verification against Ultralytics (same ONNX model, same image)
#Image: 0000242_02400_d_0000009 (960×540), imgsz 640, conf 0.25, NMS IoU 0.7

#- Detections: 26 (own pipeline) vs 26 (Ultralytics)
#- Matched: 26/26, lowest IoU 1.0, max confidence difference 0.0
#  → own preprocessing, postprocessing and NMS reproduce Ultralytics exactly.
#- Sanity test: shifting own boxes by 5 px → 17/26 matched, lowest IoU 0.51
#  → the comparison is sensitive; the perfect match is real, not a test artifact.