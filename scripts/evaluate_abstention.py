import numpy as np
import torch

from src.data import CLASS_NAMES, create_dataloaders
from src.model import build_model, get_device


HEALTHY_INDEX = CLASS_NAMES.index("healthy")
CONFIDENCE_THRESHOLD = 0.90

MARGINS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]


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
    top1_confidences = []
    margins = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            top2_values, top2_indices = torch.topk(
                probabilities,
                k=2,
                dim=1,
            )

            top1 = top2_values[:, 0]
            top2 = top2_values[:, 1]

            y_true.extend(labels.numpy())
            y_pred.extend(
                top2_indices[:, 0].cpu().numpy()
            )
            top1_confidences.extend(
                top1.cpu().numpy()
            )
            margins.extend(
                (top1 - top2).cpu().numpy()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    top1_confidences = np.array(top1_confidences)
    margins = np.array(margins)

    print(
        "Margin | Coverage | Accepted accuracy | "
        "Diseased→Healthy | Healthy→Diseased"
    )
    print("-" * 78)

    for margin_threshold in MARGINS:
        accepted = (
            (top1_confidences >= CONFIDENCE_THRESHOLD)
            & (margins >= margin_threshold)
        )

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
            f"{margin_threshold:>6.2f} | "
            f"{coverage:>7.1%} | "
            f"{accuracy:>16.1%} | "
            f"{diseased_to_healthy:>16} | "
            f"{healthy_to_diseased:>16}"
        )


if __name__ == "__main__":
    main()
