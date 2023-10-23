import cv2
import random
import numpy as np

from albumentations import DualTransform
from typing import Dict, Any, Tuple, Iterable, Sequence, List
from albumentations.core.transforms_interface import BoxInternalType, KeypointType
from albumentations.augmentations.crops import functional as F


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


class RandomSmartCrop(DualTransform):
    # scrfd default scales
    # [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # tinface default scales
    # [0.3, 0.45, 0.6, 0.8, 1.0]
    PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    def apply(self, img: np.ndarray, x_min: int = 0, x_max: int = 0, y_min: int = 0, y_max: int = 0, **params) -> np.ndarray:
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()

    def __init__(self, always_apply=False, p=0.5):
        super(RandomSmartCrop, self).__init__(always_apply=always_apply, p=p)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img_h, img_w = params["image"].shape[:2]
        boxes = params["bboxes"]

        l, t, r, b = 0, 0, img_w - 1, img_h - 1

        if len(boxes) == 0:
            return {'x_min': l, 'y_min': t, 'x_max': r, 'y_max': b}

        boxes = np.stack([b[:4] for b in boxes])
        boxes[:, [0, 2]] *= img_w
        boxes[:, [1, 3]] *= img_h

        for _ in range(250):
            scale = random.choice(self.PRE_SCALES)
            short_side = min(img_w, img_h)
            crop_w = int(scale * short_side)
            crop_h = crop_w

            l = random.randint(0, abs(img_w - crop_w))
            t = random.randint(0, abs(img_h - crop_h))
            r, b = l + crop_w, t + crop_h
            roi = np.array((l, t, r, b))

            value = matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue
            else:
                break

        return {'x_min': l, 'y_min': t, 'x_max': r, 'y_max': b}

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_crop(bbox, **params)

    def apply_to_keypoint(
        self,
        keypoint: Tuple[float, float, float, float],
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        **params
    ) -> Tuple[float, float, float, float]:
        return F.crop_keypoint_by_coords(keypoint, crop_coords=(x_min, y_min, x_max, y_max))
