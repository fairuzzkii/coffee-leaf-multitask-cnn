import torch
import torch.nn as nn
import torchvision.models as tv_models


__all__ = ["EfficientNetB0", "efficientnet", "efficientnet_b0"]


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 multi-task architecture for disease classification and severity estimation."""

    def __init__(self, pretrained=False, num_classes=(5, 5)):
        super(EfficientNetB0, self).__init__()

        # Load pretrained weights if requested using torchvision models
        if pretrained:
            try:
                weights = tv_models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = tv_models.efficientnet_b0(weights=weights)
            except Exception:
                self.backbone = tv_models.efficientnet_b0(pretrained=True)
        else:
            try:
                self.backbone = tv_models.efficientnet_b0(weights=None)
            except Exception:
                self.backbone = tv_models.efficientnet_b0(pretrained=False)

        # Get feature dimension before replacing classifier (1280 for EfficientNet-B0)
        if hasattr(self.backbone.classifier, "__getitem__"):
            self.last_channel = self.backbone.classifier[1].in_features
        else:
            self.last_channel = 1280

        # Replace default classifier with nn.Identity() to output feature vector
        self.backbone.classifier = nn.Identity()

        # Build classification heads matching existing architecture pattern (e.g. MobileNetV2)
        if type(num_classes) is not tuple:
            # Single task
            self.classifier1 = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes)
            )
            self.classifier2 = None
        else:
            # Multi task: head 1 for disease classification, head 2 for severity estimation
            self.classifier1 = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes[0])
            )
            self.classifier2 = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes[1])
            )

        # Initialize head weights matching project convention
        for head in [self.classifier1, self.classifier2]:
            if head is not None:
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(x)

        if self.classifier2 is None:
            x1 = self.classifier1(feat)
            return x1
        else:
            x1 = self.classifier1(feat)
            x2 = self.classifier2(feat)
            return x1, x2


def efficientnet_b0(pretrained=False, **kwargs):
    """Constructs an EfficientNet-B0 model."""
    return EfficientNetB0(pretrained=pretrained, **kwargs)


def efficientnet(pretrained=False, **kwargs):
    """Alias for efficientnet_b0."""
    return EfficientNetB0(pretrained=pretrained, **kwargs)
