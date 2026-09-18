from torchvision import transforms

IMAGE_SIZE = 224

CLASS_NAMES = [
    "angular_leaf_spot",
    "bean_rust",
    "healthy",
]

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
