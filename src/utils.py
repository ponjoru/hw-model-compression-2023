import time
import torch
import numpy as np
import albumentations as A
import albumentations_experimental as AE

from albumentations.pytorch import ToTensorV2

from src.transforms import RandomSmartCrop


def compute_pytorch_latency(model, input_size, n_iters=150, device='cpu'):
    assert n_iters > 10

    model.eval().to(device)
    x = torch.rand(input_size).to(device)

    time_arr = []
    with torch.no_grad():
        for i in range(n_iters):
            t0 = time.time()
            _ = model(x)
            t1 = time.time()

            if i > 9:
                time_arr.append(t1-t0)

    t = sum(time_arr) / len(time_arr) * 1000
    return t


def compute_pipeline_latency(pipeline, sample):
    times_arr = []
    for i in range(100):
        t0 = time.time()
        _ = pipeline(sample)
        t1 = time.time()

        if i > 10:
            times_arr.append(t1-t0)
    return sum(times_arr) / len(times_arr) * 1000


def compute_numpy_latency(model, input_size, n_iters=150):
    assert n_iters > 10
    x = np.random.rand(*input_size).astype(np.float32)

    time_arr = []
    for i in range(n_iters):
        t0 = time.time()
        _ = model.forward(x)
        t1 = time.time()

        if i > 9:
            time_arr.append(t1-t0)

    t = sum(time_arr) / len(time_arr) * 1000
    return t



def get_transforms(mode):
    if mode == 'train':
        t = A.Compose([
                RandomSmartCrop(p=1),
                A.LongestMaxSize(max_size=640),
                A.PadIfNeeded(min_height=640, min_width=640, value=[0, 0, 0], border_mode=0, position='top_left'),
                A.ColorJitter(hue=0.0705, saturation=[0.5, 1.5], contrast=[0.5, 1.5], brightness=0.1254, p=0.3),
                AE.HorizontalFlipSymmetricKeypoints(symmetric_keypoints=[[0, 1], [2, 2], [3, 4]], p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format='coco', min_visibility=0.7, label_fields=['bb_classes', 'bb_ignore', 'bb_id']),
            keypoint_params=A.KeypointParams(format='xy', label_fields=['kp_classes', 'kp2bb_id', 'kp_ignore'], remove_invisible=False)
        )
    else:
        t = A.Compose([
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(min_height=640, min_width=640, value=[0, 0, 0], border_mode=0, position='top_left'),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196]),
            ToTensorV2(),
        ],
            bbox_params=A.BboxParams(format='coco', min_visibility=0.7,
                                     label_fields=['bb_classes', 'bb_ignore', 'bb_id']),
            keypoint_params=A.KeypointParams(format='xy', label_fields=['kp_classes', 'kp2bb_id', 'kp_ignore'],
                                             remove_invisible=True)
        )
    return t
