"""Simulate Unity C# preprocessing and output parsing for the Unity AR team.

This script demonstrates exactly how Unity Sentis should:
1. Preprocess an input camera image (Resize 224x224, RGB, ImageNet normalization).
2. Run inference on the exported ONNX model.
3. Parse the 3 outputs:
   - disease_probs: argmax -> disease label + confidence
   - severity: continuous ratio -> percentage (0 - 100%)
   - bbox: model [cx, cy, w, h] -> Unity viewport [cx, y_unity, w, h]
4. Export the parsed result to a sample JSON for the Unity team.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image


def simulate_unity_pipeline(
    image_path,
    onnx_path="results/model_unity.onnx",
    meta_path="results/model_unity_meta.json",
    output_json="results/sample_unity_prediction.json",
):
    print("=======================================================")
    print("Simulating Unity Sentis Inference & Parsing Pipeline")
    print("=======================================================")

    # 1. Load Metadata
    label_map = {
        0: "Healthy",
        1: "Miner",
        2: "Leaf rust",
        3: "Phoma",
        4: "Cercospora",
    }
    if os.path.exists(meta_path):
        with open(meta_path, "r") as fp:
            meta = json.load(fp)
            raw_labels = meta.get("outputs", {}).get("disease_probs", {}).get("labels", {})
            label_map = {int(k): v for k, v in raw_labels.items()}
            print(f"Loaded label map from metadata: {label_map}")

    # 2. Preprocessing (Exact steps Unity Sentis / C# Texture2D should do)
    # A. Open image as RGB
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"Original image size: {orig_w}x{orig_h}")

    # B. Direct resize to 224x224 (no crop, as specified in Section 3.4)
    img_resized = img.resize((224, 224), Image.BILINEAR)

    # C. Convert to float32 [0.0, 1.0]
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # D. Standard ImageNet Normalization: (x - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_np - mean) / std

    # E. Change shape from [H, W, C] to NCHW format: [1, 3, 224, 224]
    input_tensor = np.transpose(img_norm, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 3. ONNX Runtime Inference
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed. Install via: pip install onnxruntime")
        return None

    if not os.path.exists(onnx_path):
        print(f"ONNX model file not found at: {onnx_path}")
        print("Please run export_onnx.py first.")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: input_tensor})
    # outputs[0] = disease_probs [1, 5]
    # outputs[1] = severity [1, 1]
    # outputs[2] = bbox [1, 4]
    disease_probs = outputs[0][0]
    severity_val = float(outputs[1][0][0])
    bbox_val = outputs[2][0]

    # 4. Parsing for Unity AR
    predicted_class_id = int(np.argmax(disease_probs))
    predicted_disease_name = label_map.get(predicted_class_id, "Unknown")
    confidence = float(disease_probs[predicted_class_id])

    severity_percentage = float(severity_val * 100.0)

    # Model coordinate: origin top-left (x right, y down)
    cx_model, cy_model, w_model, h_model = (
        float(bbox_val[0]),
        float(bbox_val[1]),
        float(bbox_val[2]),
        float(bbox_val[3]),
    )

    # Unity viewport coordinate conversion (y flipped: origin bottom-left)
    y_unity = 1.0 - cy_model

    parsed_result = {
        "source_image": os.path.basename(image_path),
        "diagnosis": {
            "predicted_class_id": predicted_class_id,
            "disease_name": predicted_disease_name,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                label_map[i]: round(float(disease_probs[i]), 4) for i in range(len(disease_probs))
            },
        },
        "severity": {
            "raw_ratio": round(severity_val, 4),
            "percentage": f"{severity_percentage:.1f}%",
        },
        "localization": {
            "model_coordinates_top_left_origin": {
                "center_x": round(cx_model, 4),
                "center_y": round(cy_model, 4),
                "width": round(w_model, 4),
                "height": round(h_model, 4),
            },
            "unity_viewport_coordinates_bottom_left_origin": {
                "center_x": round(cx_model, 4),
                "center_y": round(y_unity, 4),
                "width": round(w_model, 4),
                "height": round(h_model, 4),
            },
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as fp:
        json.dump(parsed_result, fp, indent=2)

    print("\n--- Unity AR Parsed Output Sample ---")
    print(json.dumps(parsed_result, indent=2))
    print(f"\nSaved sample JSON output for Unity team: {output_json}")
    return parsed_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Unity Preprocessing & Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to coffee leaf test image")
    parser.add_argument("--onnx", type=str, default="results/model_unity.onnx")
    parser.add_argument("--meta", type=str, default="results/model_unity_meta.json")
    parser.add_argument("--output", type=str, default="results/sample_unity_prediction.json")

    args = parser.parse_args()
    simulate_unity_pipeline(args.image, args.onnx, args.meta, args.output)
