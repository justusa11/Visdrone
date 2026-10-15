import cv2
import numpy as np
import onnxruntime as ort


# --------------------------------------------------
# IoU
# --------------------------------------------------

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_left = max(ax1, bx1)
    inter_top = max(ay1, by1)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)

    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)

    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# NMS
# --------------------------------------------------

def nms(
    boxes,
    scores,
    classes,
    iou_threshold=0.7
):
    keep = []

    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]

        indices = sorted(
            indices,
            key=lambda i: scores[i],
            reverse=True
        )

        while indices:
            best = indices.pop(0)
            keep.append(best)

            remaining = []

            for idx in indices:
                overlap = iou(
                    boxes[best],
                    boxes[idx]
                )

                if overlap <= iou_threshold:
                    remaining.append(idx)

            indices = remaining

    return keep


# --------------------------------------------------
# Preprocess
# --------------------------------------------------

def preprocess(image, input_size=640):
    orig_h, orig_w = image.shape[:2]

    scale = min(
        input_size / orig_w,
        input_size / orig_h
    )

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2

    pad_right = input_size - new_w - pad_x
    pad_bottom = input_size - new_h - pad_y

    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        pad_bottom,
        pad_x,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    tensor = rgb.astype(np.float32) / 255.0

    tensor = np.transpose(
        tensor,
        (2, 0, 1)
    )

    tensor = np.expand_dims(
        tensor,
        axis=0
    )

    tensor = np.ascontiguousarray(
        tensor,
        dtype=np.float32
    )

    return tensor, scale, pad_x, pad_y


# --------------------------------------------------
# Inference
# --------------------------------------------------

def infer(session, tensor):
    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: tensor}
    )

    return outputs[0]


# --------------------------------------------------
# Postprocess
# --------------------------------------------------

def postprocess(
    raw_output,
    scale,
    pad_x,
    pad_y,
    orig_w,
    orig_h,
    conf_threshold=0.25,
    iou_threshold=0.7
):
    pred = raw_output[0].T

    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1
    )

    scores = np.max(
        class_scores,
        axis=1
    )

    mask = scores >= conf_threshold

    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]

    boxes_xyxy = []

    for cx, cy, w, h in boxes_cxcywh:
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes_xyxy.append(
            [x1, y1, x2, y2]
        )

    boxes_xyxy = np.array(
        boxes_xyxy,
        dtype=np.float32
    )

    if len(boxes_xyxy) == 0:
        return []

    keep_indices = nms(
        boxes_xyxy,
        scores,
        class_ids,
        iou_threshold=iou_threshold
    )

    boxes_xyxy = boxes_xyxy[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]

    detections = []

    for box, score, class_id in zip(
        boxes_xyxy,
        scores,
        class_ids
    ):
        x1, y1, x2, y2 = box

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        detections.append(
            {
                "class_id": int(class_id),
                "confidence": float(score),
                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            }
        )

    return detections

#--------------------------------------------------
# DEMO
#--------------------------------------------------
if __name__ == "__main__":

    model_path = "models/visdrone_yolo11n_640.onnx"
    image_path = "image/0000242_02400_d_0000009.jpg"

    # Load ONNX model
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    orig_h, orig_w = image.shape[:2]

    # Preprocess
    tensor, scale, pad_x, pad_y = preprocess(
        image,
        input_size=640
    )

    # Inference
    raw_output = infer(
        session,
        tensor
    )

    # Postprocess
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

    print(
        "Final detections:",
        len(detections)
    )

    detections_sorted = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    for det in detections_sorted[:10]:
        print(det)

    class_names = [
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor"
    ]

    debug_image = image.copy()

    for det in detections:
        class_id = det["class_id"]
        confidence = det["confidence"]

        x1, y1, x2, y2 = map(
            int,
            det["box"]
        )

        label = (
            f"{class_names[class_id]} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            debug_image,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(
        "onnx_detections_refactored.jpg",
        debug_image
    )

    print(
        "Saved onnx_detections_refactored.jpg"
    )