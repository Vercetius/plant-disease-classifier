from pathlib import Path

import torch
from PIL import Image

from src.model import build_model, get_device
from src.preprocessing import CLASS_NAMES, eval_transform


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "plant_disease_model.pth"

CONFIDENCE_THRESHOLD = 0.70


def load_trained_model():
    device = get_device()

    model = build_model(
        freeze_features=False,
        pretrained=False,
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    model = model.to(device)
    model.eval()

    return model, device


def predict_image(
    image: Image.Image,
    model,
    device,
):
    image = image.convert("RGB")

    tensor = (
        eval_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    values, indices = torch.sort(
        probabilities,
        descending=True,
    )

    predictions = []

    for probability, index in zip(
        values.cpu().numpy(),
        indices.cpu().numpy(),
    ):
        predictions.append(
            {
                "class": CLASS_NAMES[index],
                "confidence": float(probability),
            }
        )

    top_prediction = predictions[0]
    confidence = top_prediction["confidence"]

    return {
        "prediction": top_prediction["class"],
        "confidence": confidence,
        "accepted": confidence >= CONFIDENCE_THRESHOLD,
        "top_3": predictions,
    }
