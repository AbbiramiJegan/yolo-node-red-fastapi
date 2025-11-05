from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np, cv2, onnxruntime as ort, time, os
from tempfile import NamedTemporaryFile
from PIL import Image

app = FastAPI(title="YOLO ONNX Inference API")

# ---------- CONFIG ----------
MODEL_PATH = os.path.expanduser(r"C:\Users\User\.node-red\scripts\yolo11n.onnx")

# ---------- LOAD MODEL ONCE ----------
providers = ["CPUExecutionProvider"]
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# ---------- PREPROCESS ----------
def preprocess_image(file_path: str, img_size=640):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Cannot read image: {file_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    r = img_size / max(h0, w0)
    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[: img.shape[0], : img.shape[1]] = img
    blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, (h0, w0)

# ---------- COCO DATASET -------

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# ---------- INFERENCE ----------
def run_inference(image_path: str):
    blob, (h0, w0) = preprocess_image(image_path)
    start = time.time()
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    infer_time = (time.time() - start) * 1000
    return outputs, infer_time

def postprocess_yolo_output(outputs, conf_threshold=0.25, iou_threshold=0.45):
    preds = outputs[0][0]  # shape: (84, 8400)
    boxes = preds[:4, :].T  # xywh → convert later
    scores = preds[4:, :].T  # class scores for each box

    confidences = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    # Filter by confidence threshold
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convert xywh → xyxy (top-left, bottom-right)
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    results = []
    for i in range(len(confidences)):
        results.append({
            "class": COCO_CLASSES[class_ids[i]],
            "confidence": round(float(confidences[i]), 3),
            "box": xyxy_boxes[i].tolist()
        })

    return results


# ---------- ROUTES ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        outputs, infer_time = run_inference(tmp_path)
        detections = postprocess_yolo_output(outputs)
        os.remove(tmp_path)

        # Pick detection with highest confidence
        if detections:
            best_detection = max(detections, key=lambda x: x["confidence"])
        else:
            best_detection = None

        return JSONResponse({
            "success": True,
            "best_detection": best_detection,
            "inference_time_ms": round(infer_time, 2)
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.get("/")
def home():
    return {"message": "YOLO ONNX FastAPI server is running."}