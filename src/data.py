from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader


IMAGE_SIZE = 224

CLASS_NAMES = [
    "angular_leaf_spot",
    "bean_rust",
    "healthy",
]


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0),
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class BeansDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image = item["image"].convert("RGB")
        label = item["labels"]

        image = self.transform(image)

        return image, label


def load_beans_dataset():
    return load_dataset("AI-Lab-Makerere/beans")


def create_dataloaders(batch_size=32):
    dataset = load_beans_dataset()

    train_dataset = BeansDataset(
        dataset["train"],
        train_transform,
    )

    validation_dataset = BeansDataset(
        dataset["validation"],
        eval_transform,
    )

    test_dataset = BeansDataset(
        dataset["test"],
        eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )
