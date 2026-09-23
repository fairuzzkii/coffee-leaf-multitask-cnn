import torch
from architectures import cnn_model


def test_efficientnet_multitask():
    print("--- Testing EfficientNet Multitask (pretrained=True) ---")
    model = cnn_model("efficientnet", pretrained=True, num_classes=(5, 5))
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        out_dis, out_sev = model(dummy_input)

    print(f"Output Disease shape : {out_dis.shape}")
    print(f"Output Severity shape: {out_sev.shape}")

    assert out_dis.shape == torch.Size([2, 5]), f"Expected (2, 5), got {out_dis.shape}"
    assert out_sev.shape == torch.Size([2, 5]), f"Expected (2, 5), got {out_sev.shape}"
    print("Multi-task forward pass: SUCCESS\n")


def test_efficientnet_singletask():
    print("--- Testing EfficientNet Single-Task (pretrained=False) ---")
    model = cnn_model("efficientnet", pretrained=False, num_classes=5)
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        out = model(dummy_input)

    print(f"Output shape: {out.shape}")
    assert out.shape == torch.Size([2, 5]), f"Expected (2, 5), got {out.shape}"
    print("Single-task forward pass: SUCCESS\n")


def test_mobilenet_v2_regression():
    print("--- Testing MobileNetV2 (Regression check) ---")
    model = cnn_model("mobilenet_v2", pretrained=False, num_classes=(5, 5))
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        out_dis, out_sev = model(dummy_input)

    print(f"MobileNet out_dis shape: {out_dis.shape}")
    print(f"MobileNet out_sev shape: {out_sev.shape}")

    assert out_dis.shape == torch.Size([2, 5])
    assert out_sev.shape == torch.Size([2, 5])
    print("MobileNetV2 check: SUCCESS\n")


if __name__ == "__main__":
    test_efficientnet_multitask()
    test_efficientnet_singletask()
    test_mobilenet_v2_regression()
    print("All sanity checks passed successfully!")
