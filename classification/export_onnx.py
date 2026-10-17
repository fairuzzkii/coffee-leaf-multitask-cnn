"""Export trained MultiTaskCoffeeNet to ONNX format with exact named outputs for Unity Sentis.

Adheres strictly to Section 3.6 of PROJECT_CONTEXT.md.
Outputs:
  - disease_probs: [1, N_kelas] (softmax applied)
  - severity: [1, 1] (continuous 0.0 - 1.0)
  - bbox: [1, 4] (cx, cy, w, h in 0.0 - 1.0)
"""

import os
import json
import argparse
import datetime
import torch

from architectures.multitask_model import build_model


def export_to_onnx(
    weights_path,
    output_onnx_path,
    backbone="mobilenet",
    num_disease_classes=5,
    opset_version=12,
):
    print(f"Exporting model to ONNX...")
    print(f"Weights: {weights_path}")
    print(f"Backbone: {backbone}")
    print(f"Output ONNX: {output_onnx_path}")

    device = torch.device("cpu")
    model = build_model(
        backbone=backbone,
        mode="multi",
        num_disease_classes=num_disease_classes,
        pretrained=False,
    )

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Successfully loaded model weights.")
    else:
        print("Warning: Weights file not found or not specified. Exporting model with random weights for testing graph.")

    model.eval()
    model.set_export_mode(True)  # Applies Softmax on disease head

    # Dummy input: [1, 3, 224, 224] RGB float32
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    input_names = ["input"]
    output_names = ["disease_probs", "severity", "bbox"]

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # Fixed shape [1, 3, 224, 224] optimal for mobile / Unity Sentis
    )

    print(f"ONNX model successfully saved to: {output_onnx_path}")

    # Generate model_meta.json
    meta_path = os.path.splitext(output_onnx_path)[0] + "_meta.json"
    label_map = {
        0: "Healthy",
        1: "Miner",
        2: "Leaf rust",
        3: "Phoma",
        4: "Cercospora",
    }

    metadata = {
        "model_name": f"coffee_leaf_multitask_{backbone}",
        "backbone": backbone,
        "export_date": datetime.datetime.now().isoformat(),
        "opset_version": opset_version,
        "input": {
            "name": "input",
            "shape": [1, 3, 224, 224],
            "dtype": "float32",
            "color_format": "RGB",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "formula": "(pixel / 255.0 - mean) / std",
            },
        },
        "outputs": {
            "disease_probs": {
                "shape": [1, num_disease_classes],
                "dtype": "float32",
                "description": "Softmax class probabilities. Predicted class is argmax.",
                "labels": label_map,
            },
            "severity": {
                "shape": [1, 1],
                "dtype": "float32",
                "unit": "normalized_ratio",
                "range": [0.0, 1.0],
                "description": "Continuous infection severity ratio. Multiply by 100 to get percentage.",
            },
            "bbox": {
                "shape": [1, 4],
                "dtype": "float32",
                "format": "[center_x, center_y, width, height]",
                "range": [0.0, 1.0],
                "coordinate_convention": {
                    "model_origin": "top-left (x right, y down)",
                    "unity_conversion_note": "Unity UI/viewport origin is bottom-left. Use: y_unity = 1.0 - y_model",
                },
            },
        },
    }

    with open(meta_path, "w") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Metadata file successfully saved to: {meta_path}")
    return output_onnx_path, meta_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MultiTask Coffee Model to ONNX for Unity Sentis")
    parser.add_argument("--weights", type=str, default="results/multitask_3head_mobilenet/best_weights.pth")
    parser.add_argument("--output", type=str, default="results/model_unity.onnx")
    parser.add_argument("--backbone", type=str, default="mobilenet", choices=["mobilenet", "efficientnet"])
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--opset", type=int, default=12)

    args = parser.parse_args()
    export_to_onnx(
        weights_path=args.weights,
        output_onnx_path=args.output,
        backbone=args.backbone,
        num_disease_classes=args.num_classes,
        opset_version=args.opset,
    )
