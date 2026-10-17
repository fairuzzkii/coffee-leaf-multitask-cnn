import torch
import torch.nn as nn
import torchvision.models as tv_models


class MultiTaskCoffeeNet(nn.Module):
    """Multi-Task CNN model for Coffee Leaf Analysis.

    Outputs (Multi-Task Mode):
        1. disease: classification logits (or softmax probabilities during export)
        2. severity: continuous regression in range [0, 1] (via Sigmoid)
        3. bbox: normalized bounding box [cx, cy, w, h] in range [0, 1] (via Sigmoid)

    Supports both MobileNetV2 and EfficientNet-B0 backbones.
    """

    def __init__(
        self,
        backbone_name="mobilenet",
        num_disease_classes=5,
        pretrained=True,
        mode="multi",
        single_task="disease",
        dropout_rate=0.2,
    ):
        super(MultiTaskCoffeeNet, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.num_disease_classes = num_disease_classes
        self.mode = mode.lower()
        self.single_task = single_task.lower()
        self.export_mode = False

        # 1. Initialize Backbone
        if "mobilenet" in self.backbone_name:
            if pretrained:
                try:
                    weights = tv_models.MobileNet_V2_Weights.DEFAULT
                    base = tv_models.mobilenet_v2(weights=weights)
                except Exception:
                    base = tv_models.mobilenet_v2(pretrained=True)
            else:
                try:
                    base = tv_models.mobilenet_v2(weights=None)
                except Exception:
                    base = tv_models.mobilenet_v2(pretrained=False)

            self.feature_extractor = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.in_features = 1280

        elif "efficientnet" in self.backbone_name:
            if pretrained:
                try:
                    weights = tv_models.EfficientNet_B0_Weights.DEFAULT
                    base = tv_models.efficientnet_b0(weights=weights)
                except Exception:
                    base = tv_models.efficientnet_b0(pretrained=True)
            else:
                try:
                    base = tv_models.efficientnet_b0(weights=None)
                except Exception:
                    base = tv_models.efficientnet_b0(pretrained=False)

            self.feature_extractor = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.in_features = 1280
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. Choose 'mobilenet' or 'efficientnet'."
            )

        # 2. Build Task Heads
        if self.mode == "multi":
            # Disease Classification Head
            self.disease_head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.in_features, self.num_disease_classes),
            )
            # Severity Regression Head (output 1 continuous value, bounded to [0, 1])
            self.severity_head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.in_features, 1),
                nn.Sigmoid(),
            )
            # Localization Bounding Box Head (output 4 values: cx, cy, w, h bounded to [0, 1])
            self.bbox_head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.in_features, 4),
                nn.Sigmoid(),
            )
            self._init_head_weights(self.disease_head)
            self._init_head_weights(self.severity_head)
            self._init_head_weights(self.bbox_head)

        elif self.mode == "single":
            if self.single_task == "disease":
                self.head = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(self.in_features, self.num_disease_classes),
                )
            elif self.single_task == "severity":
                self.head = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(self.in_features, 1),
                    nn.Sigmoid(),
                )
            elif self.single_task in ["bbox", "localization"]:
                self.head = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(self.in_features, 4),
                    nn.Sigmoid(),
                )
            else:
                raise ValueError(f"Unknown single task: {single_task}")
            self._init_head_weights(self.head)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'multi' or 'single'.")

    def _init_head_weights(self, head):
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_export_mode(self, enabled=True):
        """Set model to export mode (applies softmax on disease output)."""
        self.export_mode = enabled

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)

        if self.mode == "single":
            out = self.head(feat)
            if self.single_task == "disease" and self.export_mode:
                out = torch.softmax(out, dim=1)
            return out

        # Multi-task mode
        disease_out = self.disease_head(feat)
        severity_out = self.severity_head(feat)
        bbox_out = self.bbox_head(feat)

        if self.export_mode:
            # During ONNX export, apply softmax to disease logits
            disease_out = torch.softmax(disease_out, dim=1)

        return disease_out, severity_out, bbox_out


def build_model(
    backbone="mobilenet",
    mode="multi",
    num_disease_classes=5,
    pretrained=True,
    single_task="disease",
    dropout_rate=0.2,
    weights_path=None,
):
    """Factory helper to build and optionally load weights for MultiTaskCoffeeNet."""
    model = MultiTaskCoffeeNet(
        backbone_name=backbone,
        num_disease_classes=num_disease_classes,
        pretrained=pretrained,
        mode=mode,
        single_task=single_task,
        dropout_rate=dropout_rate,
    )

    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Error loading weights from {weights_path}: {e}")

    return model
