import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.optim import Adam

from src.data import (
    CLASS_NAMES,
    create_dataloaders,
)
from src.model import (
    build_model,
    get_device,
    unfreeze_last_blocks,
)


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"

MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions.extend(
            outputs.argmax(dim=1).detach().cpu().numpy()
        )

        targets.extend(
            labels.detach().cpu().numpy()
        )

    loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
    )

    return loss, accuracy, macro_f1


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions.extend(
            outputs.argmax(dim=1).cpu().numpy()
        )

        targets.extend(
            labels.cpu().numpy()
        )

    loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
    )

    return (
        loss,
        accuracy,
        macro_f1,
        np.array(targets),
        np.array(predictions),
    )


def run_phase(
    name,
    model,
    train_loader,
    validation_loader,
    optimizer,
    criterion,
    device,
    epochs,
    best_f1,
):
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            val_loss,
            val_acc,
            val_f1,
            _,
            _,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"{name} | Epoch {epoch}/{epochs} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_acc:.4f} | "
            f"Train F1 {train_f1:.4f} | "
            f"Val loss {val_loss:.4f} | "
            f"Val acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1

            torch.save(
                model.state_dict(),
                MODELS_DIR / "plant_disease_model.pth",
            )

            print(
                f"Saved new best model "
                f"(Val Macro F1: {best_f1:.4f})"
            )

    return best_f1


def main():
    print("Creating dataloaders...")

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders(
        batch_size=32,
    )

    device = get_device()

    print("Device:", device)
    print("Train images:", len(train_loader.dataset))
    print(
        "Validation images:",
        len(validation_loader.dataset),
    )
    print("Test images:", len(test_loader.dataset))
    print()

    model = build_model(
        freeze_features=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    print("PHASE 1 — Training classifier head")

    optimizer = Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=1e-3,
    )

    best_f1 = run_phase(
        name="Head",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=4,
        best_f1=0.0,
    )

    print()
    print("PHASE 2 — Fine-tuning final feature blocks")

    model = unfreeze_last_blocks(
        model,
        blocks=2,
    )

    optimizer = Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=1e-4,
    )

    best_f1 = run_phase(
        name="FineTune",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=4,
        best_f1=best_f1,
    )

    print()
    print("PHASE 3 — Full-model fine-tuning")

    for parameter in model.parameters():
        parameter.requires_grad = True

    optimizer = Adam(
        model.parameters(),
        lr=3e-5,
    )

    best_f1 = run_phase(
        name="FullTune",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=3,
        best_f1=best_f1,
    )

    print()
    print("Loading best checkpoint...")

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "plant_disease_model.pth",
            map_location=device,
            weights_only=True,
        )
    )

    (
        test_loss,
        test_accuracy,
        test_macro_f1,
        y_true,
        y_pred,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print()

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    print(report)

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    print("Confusion matrix:")
    print(matrix)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "best_validation_macro_f1": float(best_f1),
        "class_names": CLASS_NAMES,
    }

    with open(
        MODELS_DIR / "metrics.json",
        "w",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    image = ax.imshow(matrix)

    ax.set_xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        rotation=30,
        ha="right",
    )

    ax.set_yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test Confusion Matrix")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        IMAGES_DIR / "confusion-matrix.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        "Saved model:",
        MODELS_DIR / "plant_disease_model.pth",
    )

    print(
        "Saved metrics:",
        MODELS_DIR / "metrics.json",
    )

    print(
        "Saved confusion matrix:",
        IMAGES_DIR / "confusion-matrix.png",
    )


if __name__ == "__main__":
    main()
