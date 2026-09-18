import numpy as np
import torch
from PIL import Image

from src.preprocessing import eval_transform


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, device, class_index=None):
        self.model.zero_grad(set_to_none=True)

        tensor = eval_transform(
            image.convert("RGB")
        ).unsqueeze(0).to(device)

        output = self.model(tensor)

        if class_index is None:
            class_index = output.argmax(dim=1).item()

        score = output[0, class_index]
        score.backward()

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * self.activations
        ).sum(dim=1)

        cam = torch.relu(cam)[0]

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), class_index

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_gradcam(image, cam):
    import matplotlib.pyplot as plt

    image = image.convert("RGB")
    original = np.asarray(image).astype(np.float32) / 255.0

    height, width = original.shape[:2]

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    cam_resized = np.asarray(cam_image).astype(np.float32) / 255.0

    heatmap = plt.get_cmap("jet")(cam_resized)[..., :3]

    overlay = (
        0.55 * original
        + 0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1,
    )

    return np.uint8(
        overlay * 255
    )
