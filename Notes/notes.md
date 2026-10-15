### Step 4.7 — Python + ONNX Runtime benchmark (laptop CPU, 50 runs after 5 warm-up)
Image: 0000242_02400_d_0000009 (960×540)

| Input | Pre    | Infer   | Post   | Total (median) | FPS | Detections |
| 640   | 13.3   | 70.4    | 4.8    | 88.8 ms        | 11  | 26         |
| 416   |  6.3   | 34.2    | 4.6    | 45.6 ms        | 22  | 24         |
| 320   |  3.5   | 16.2    | 1.7    | 21.2 ms        | 47  | 19         |

- Inference ≈ 80% of total; it scales with input pixel count.
- 640 → 320: ~4× faster, ~27% fewer detections (small objects lost first).
- Expected C++ gain is limited to pre/post (~20%); inference engine is the same.