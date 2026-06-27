"""Grad-CAM visual explanations for trained classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from . import config
from .dataset import SolarFlareImageDataset, build_transforms
from .models import get_model
from .utils import ensure_dir, get_device, get_logger, setup_logging

logger = get_logger(__name__)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        score = logits.sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_tensor = (weights * self.activations).sum(dim=1)
        cam_tensor = F.relu(cam_tensor)
        cam_tensor = F.interpolate(
            cam_tensor.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        cam_tensor = cam_tensor - cam_tensor.min()
        cam_tensor = cam_tensor / (cam_tensor.max() + 1e-8)
        return cam_tensor.detach().cpu().numpy()


def resolve_target_layer(model: nn.Module, model_name: str | None = None) -> nn.Module:
    name = (model_name or "").lower()
    if name == "resnet18" and hasattr(model, "layer4"):
        return model.layer4[-1]
    if name in {"efficientnet_b0", "convnext_tiny"} and hasattr(model, "features"):
        return model.features[-1]

    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("Could not find a convolutional layer for Grad-CAM.")


def overlay_cam(image_path: str | Path, cam_array: np.ndarray, output_path: str | Path, alpha: float = 0.45) -> None:
    base = Image.open(image_path).convert("RGB").resize((cam_array.shape[1], cam_array.shape[0]))
    base_array = np.asarray(base).astype(np.float32) / 255.0
    heatmap = cm.get_cmap("jet")(cam_array)[..., :3]
    overlay = np.clip((1 - alpha) * base_array + alpha * heatmap, 0, 1)
    output = Image.fromarray((overlay * 255).astype(np.uint8))
    output.save(output_path)


def generate_gradcam(args: argparse.Namespace) -> None:
    setup_logging()
    logger.info("Starting Grad-CAM generation")
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or checkpoint.get("model_name", "efficientnet_b0")
    image_size = args.image_size or checkpoint.get("image_size", config.IMAGE_SIZE)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Image CSV: %s", args.csv)

    model = get_model(model_name, pretrained=False).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = SolarFlareImageDataset(
        args.csv,
        image_root=args.image_root,
        transform=build_transforms(image_size, train=False),
        image_col=args.image_col,
        path_col=args.path_col,
        label_col=args.label_col,
        channel=args.channel,
        return_path=True,
    )

    output_dir = ensure_dir(args.output_dir)
    target_layer = resolve_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    logger.info("Loaded %d samples for Grad-CAM", len(dataset))
    logger.info("Grad-CAM output directory: %s", output_dir)

    try:
        for index in range(min(args.limit, len(dataset))):
            image, label, image_path = dataset[index]
            image = image.unsqueeze(0).to(device)
            cam_array = gradcam(image)
            probability = torch.sigmoid(model(image).view(-1)).item()
            prediction = int(probability >= args.threshold)
            output_path = output_dir / f"{model_name}_{index:04d}_y{int(label.item())}_p{prediction}.png"
            overlay_cam(image_path, cam_array, output_path)
            logger.info("Saved Grad-CAM image %s", output_path)
    finally:
        gradcam.remove_hooks()

    logger.info("Grad-CAM images saved to %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays.")
    parser.add_argument("--csv", default=str(config.LABELS_DIR / "test_labels.csv"))
    parser.add_argument("--image-root", default=str(config.SDOBENCHMARK_DIR))
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "efficientnet_b0_best.pth"))
    parser.add_argument("--model", default=None, choices=(*config.MODEL_NAMES, None))
    parser.add_argument("--channel", default="hmi")
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--path-col", default=config.IMAGE_PATH_COLUMN)
    parser.add_argument("--label-col", default=config.LABEL_COLUMN)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--threshold", type=float, default=config.PREDICTION_THRESHOLD)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output-dir", default=str(config.GRADCAM_DIR))
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    generate_gradcam(args)


if __name__ == "__main__":
    main()
