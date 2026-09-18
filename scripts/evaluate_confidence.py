import numpy as np
import torch

from src.data import CLASS_NAMES, create_dataloaders
from src.model import build_model, get_device


HEALTHY_INDEX = CLASS_NAMES.index("healthy")


def main():
    _, validation_loader, _ = create_dataloaders(
        batch_size=32
    )

    device = get_device()

    model = build_model(
        freeze_features=False
    ).to(device)

    model.load_state_dict(
        torch.load(
            "models/plant_disease_model.pth",
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    y_true = []
    y_pred = []
    confidences = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = probabilities.max(
                dim=1
            )

            y_true.extend(labels.numpy())
            y_pred.extend(
                prediction.cpu().numpy()
            )
            confidences.extend(
                confidence.cpu().numpy()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    print(
        "Threshold | Coverage | Accepted accuracy | "
        "Diseased→Healthy | Healthy→Diseased"
    )
    print("-" * 78)

    for threshold in thresholds:
        accepted = confidences >= threshold

        if accepted.sum() == 0:
            continue

        true = y_true[accepted]
        pred = y_pred[accepted]

        accuracy = (
            true == pred
        ).mean()

        diseased_to_healthy = (
            (true != HEALTHY_INDEX)
            & (pred == HEALTHY_INDEX)
        ).sum()

        healthy_to_diseased = (
            (true == HEALTHY_INDEX)
            & (pred != HEALTHY_INDEX)
        ).sum()

        coverage = accepted.mean()

        print(
            f"{threshold:>9.2f} | "
            f"{coverage:>7.1%} | "
            f"{accuracy:>16.1%} | "
            f"{diseased_to_healthy:>16} | "
            f"{healthy_to_diseased:>16}"
        )


if __name__ == "__main__":
    main()
