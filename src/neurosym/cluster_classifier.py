"""Phase 6.1.4 — inference wrapper for the size_band ResNet-18 classifier.

Loads `best.pt` once, exposes `predict(storey_name, world_xy_mm) → (band, confidence)`.
Cropping is shared with the training pipeline (10_build_cluster_crops.py) so train
and inference operate on identical patches.

Scope (mirrors F4): the caller supplies world_xy in IFC mm — at eval time that
comes from the GT element's centroid (oracle), in production from OpenCV F4's
detected opening pixel (future work).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Lazy imports for torch/PIL — keep this module importable in non-ML contexts.
_TORCH = None
_TRANSFORMS = None


def _lazy_torch():
    global _TORCH, _TRANSFORMS
    if _TORCH is None:
        import torch  # type: ignore
        from torchvision import transforms  # type: ignore
        _TORCH = torch
        _TRANSFORMS = transforms
    return _TORCH, _TRANSFORMS


CROP_PIXELS = 192
WORLD_UNIT_TO_BBOX_UNIT = 1.0 / 1000.0  # element_index mm → calibration metres


@dataclass(frozen=True)
class BandPrediction:
    band: str
    confidence: float
    logits: Tuple[float, ...]


class SizeBandClassifier:
    """ResNet-18 inference wrapper.

    Holds a cached PIL.Image per storey for fast repeated cropping.
    """

    def __init__(
        self,
        checkpoint: Path,
        calibration: Path,
        floorplans_root: Path,
        device: Optional[str] = None,
    ):
        torch, transforms = _lazy_torch()
        from torchvision.models import resnet18  # type: ignore

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        classes = ckpt.get("classes")
        if not classes:
            raise ValueError(f"checkpoint {checkpoint} missing 'classes' list")
        self.classes: List[str] = list(classes)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = resnet18(weights=None)
        model.fc = __import__("torch").nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        self.model = model

        cal_data = json.loads(Path(calibration).read_text())
        self.calibration: Dict[str, dict] = {
            entry["storey_name"]: entry for entry in cal_data.get("storeys", [])
        }
        self.floorplans_root = Path(floorplans_root)
        self.calibration_path = Path(calibration)

        self.tx = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def supported_storeys(self) -> List[str]:
        return sorted(self.calibration)

    @lru_cache(maxsize=8)
    def _load_storey_image(self, storey_name: str):
        from PIL import Image  # type: ignore

        cal = self.calibration[storey_name]
        png_rel = cal["png_path"]
        png_abs = (self.floorplans_root / png_rel).resolve()
        if not png_abs.exists():
            png_abs = (self.calibration_path.parent / Path(png_rel).name).resolve()
        return Image.open(png_abs).convert("RGB")

    def _world_to_pixel(self, world_xy_mm: Tuple[float, float], storey_name: str) -> Tuple[int, int]:
        cal = self.calibration[storey_name]
        bbox = cal["world_bbox"]
        pixel_w = cal["pixel_size"]["width"]
        pixel_h = cal["pixel_size"]["height"]
        x_m = world_xy_mm[0] * WORLD_UNIT_TO_BBOX_UNIT
        y_m = world_xy_mm[1] * WORLD_UNIT_TO_BBOX_UNIT
        span_x = bbox["xmax"] - bbox["xmin"]
        span_y = bbox["ymax"] - bbox["ymin"]
        u = (x_m - bbox["xmin"]) / span_x * pixel_w
        v = pixel_h - (y_m - bbox["ymin"]) / span_y * pixel_h
        return int(round(u)), int(round(v))

    def _crop_patch(self, storey_name: str, centre_px: Tuple[int, int]):
        from PIL import Image  # type: ignore

        img = self._load_storey_image(storey_name)
        size = CROP_PIXELS
        half = size // 2
        cx, cy = centre_px
        left = cx - half
        top = cy - half
        right = left + size
        bottom = top + size
        if left < 0 or top < 0 or right > img.width or bottom > img.height:
            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - img.width)
            pad_bottom = max(0, bottom - img.height)
            new = Image.new(
                "RGB",
                (img.width + pad_left + pad_right, img.height + pad_top + pad_bottom),
                "white",
            )
            new.paste(img, (pad_left, pad_top))
            img = new
            left += pad_left; top += pad_top
            right += pad_left; bottom += pad_top
        return img.crop((left, top, right, bottom))

    def predict(
        self,
        storey_name: str,
        world_xy_mm: Tuple[float, float],
    ) -> Optional[BandPrediction]:
        """Predict size_band for an element at world_xy_mm on `storey_name`.

        Returns None if the storey is out-of-scope (no calibration / no render).
        """
        torch, _ = _lazy_torch()
        if storey_name not in self.calibration:
            return None
        px = self._world_to_pixel(world_xy_mm, storey_name)
        crop = self._crop_patch(storey_name, px)
        x = self.tx(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)
            top_p, top_i = prob.max(dim=1)
        idx = int(top_i.item())
        return BandPrediction(
            band=self.classes[idx],
            confidence=float(top_p.item()),
            logits=tuple(logits.squeeze(0).cpu().tolist()),
        )
