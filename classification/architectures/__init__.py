import torch

from .mobilenetv2 import mobilenet_v2
from .efficientnet import efficientnet, efficientnet_b0
from .multitask_model import MultiTaskCoffeeNet, build_model


MODELS = {
    "mobilenet": mobilenet_v2,
    "mobilenet_v2": mobilenet_v2,
    "efficientnet": efficientnet,
    "efficientnet_b0": efficientnet_b0,
}


def cnn_model(model_name, pretrained=False, num_classes=(5, 5), weights_path=None):
    model_key = model_name.lower()
    if model_key not in MODELS:
        raise ValueError(
            f"Model '{model_name}' not supported. Allowed architectures: {list(MODELS.keys())}"
        )

    model = MODELS[model_key](pretrained=pretrained, num_classes=num_classes)

    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        except Exception:
            raise Exception("Error loading weights. You must train the model first.")

    if torch.cuda.is_available():
        model.cuda()

    return model
