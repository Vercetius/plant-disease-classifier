import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

NUM_CLASSES = 3


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_model(
    freeze_features=True,
    pretrained=True,
):
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    return model


def unfreeze_last_blocks(
    model,
    blocks=2,
):
    for parameter in model.features[-blocks:].parameters():
        parameter.requires_grad = True
