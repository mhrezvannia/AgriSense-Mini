from __future__ import annotations

from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(_module: torch.nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_tensor)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    heatmap_image = cm.get_cmap("YlOrRd")(heatmap)[..., :3]
    heatmap_uint8 = (heatmap_image * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(image.size)
    return Image.blend(image.convert("RGB"), heatmap_pil, alpha=alpha)
