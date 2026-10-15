import cv2
import numpy as np

from inspect_onnx import (
    preprocess,
    infer,
    postprocess,
    iou
)

image = cv2.imread(
    "image/0000242_02400_d_0000009.jpg"
)

tensor, scale, pad_x, pad_y = preprocess(
    image,
    input_size=640
)

print(
    "Python checksum:",
    np.sum(tensor, dtype=np.float64)
)

print(
    "Python pixel R:",
    tensor[0, 0, 140, 0]
)

print(
    "Python pixel G:",
    tensor[0, 1, 140, 0]
)

print(
    "Python pixel B:",
    tensor[0, 2, 140, 0]
)