"""Verification script for exported ONNX model against PyTorch model.

Ensures that the ONNX model outputs (disease_probs, severity, bbox)
match PyTorch within a tight numerical tolerance.
"""

import os
import argparse
import numpy as np
import torch

from architectures.multitask_model import build_model


def verify(onnx_path, weights_path=None, backbone="mobilenet", num_classes=5):
    print(f"\n==========================================")
    print(f"Verifying ONNX model against PyTorch model")
    print(f"ONNX: {onnx_path}")
    print(f"Backbone: {backbone}")
    print(f"==========================================\n")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed. Please install via: pip install onnxruntime")
        return False

    # 1. Load PyTorch model
    device = torch.device("cpu")
    pytorch_model = build_model(
        backbone=backbone,
        mode="multi",
        num_disease_classes=num_classes,
        pretrained=False,
    )
    if weights_path and os.path.exists(weights_path):
        pytorch_model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded PyTorch weights.")

    pytorch_model.eval()
    pytorch_model.set_export_mode(True)

    # 2. Load ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    output_names = [out.name for out in ort_session.get_outputs()]
    print(f"ONNX Input: {input_name}")
    print(f"ONNX Outputs: {output_names}")

    # 3. Generate random test input [1, 3, 224, 224]
    np.random.seed(42)
    test_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np)

    # 4. PyTorch Inference
    with torch.no_grad():
        pt_disease, pt_severity, pt_bbox = pytorch_model(test_input_torch)
        pt_disease = pt_disease.numpy()
        pt_severity = pt_severity.numpy()
        pt_bbox = pt_bbox.numpy()

    # 5. ONNX Inference
    onnx_outs = ort_session.run(None, {input_name: test_input_np})
    onnx_disease, onnx_severity, onnx_bbox = onnx_outs[0], onnx_outs[1], onnx_outs[2]

    # 6. Calculate maximum differences
    diff_disease = np.max(np.abs(pt_disease - onnx_disease))
    diff_severity = np.max(np.abs(pt_severity - onnx_severity))
    diff_bbox = np.max(np.abs(pt_bbox - onnx_bbox))

    print(f"\n--- Numerical Parity Verification ---")
    print(f"Max Difference - disease_probs : {diff_disease:.6e}")
    print(f"Max Difference - severity      : {diff_severity:.6e}")
    print(f"Max Difference - bbox          : {diff_bbox:.6e}")

    tolerance = 1e-4
    all_passed = (diff_disease < tolerance) and (diff_severity < tolerance) and (diff_bbox < tolerance)

    if all_passed:
        print("\n>>> SUCCESS: All outputs match within tolerance! Model is verified and ready for Unity Sentis. <<<")
    else:
        print("\n>>> WARNING: Output difference exceeds tolerance. Check opset and operators. <<<")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="results/model_unity.onnx")
    parser.add_argument("--weights", type=str, default="results/multitask_3head_mobilenet/best_weights.pth")
    parser.add_argument("--backbone", type=str, default="mobilenet")
    parser.add_argument("--num_classes", type=int, default=5)
    args = parser.parse_args()

    verify(args.onnx, args.weights, args.backbone, args.num_classes)
