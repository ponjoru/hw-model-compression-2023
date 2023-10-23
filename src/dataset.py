import itertools
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


def simple_collate_fn(batch):
    img, bb, kp, img_id, img_path, img_shape = zip(*batch)  # transposed
    for i, b in enumerate(bb):
        b[:, 0] = i  # add target image index
    for i, k in enumerate(kp):
        k[:, 0] = i  # add target image index

    new_batch = {
        'image': torch.stack(img, 0),
        # targets
        'bb': torch.cat(bb, 0),
        'kp': torch.cat(kp, 0),
        # meta
        'img_id': img_id,
        'img_path': img_path,
        'img_shape': img_shape,
    }

    return new_batch


class SimpleCustomBatch:
    def __init__(self, data):
        img, bb, kp, img_id, img_path, img_shape = list(zip(*data))
        for i, b in enumerate(bb):
            b[:, 0] = i  # add target image index
        for i, k in enumerate(kp):
            k[:, 0] = i  # add target image index

        self.image = torch.stack(img, 0)
        self.bb = torch.cat(bb, 0)
        self.kp = torch.cat(kp, 0)
        self.img_path = img_path
        self.img_id = img_id
        self.img_shape = img_shape

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.image = self.image.pin_memory()
        self.bb = self.bb.pin_memory()
        self.kp = self.kp.pin_memory()
        return self

    def __getitem__(self, item):
        return getattr(self, item)


def collate_fn(batch):
    return SimpleCustomBatch(batch)


def is_in_image(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


class WiderFaceDataset(Dataset):
    NK = 5
    BB_CLASS_LABELS = ('Face', )
    KP_CLASS_LABELS = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

    def __init__(self, ds_path, mode, min_size=None, transforms=None, color_layout='RGB'):
        super(WiderFaceDataset, self).__init__()

        self.min_size = min_size
        self.ds_path = ds_path
        self.mode = mode
        self.transforms = transforms
        self.color_layout = color_layout

        self.gt_path = str(Path(ds_path) / mode / 'labelv2.txt')

        self.bb_cat2id = {cat: idx for idx, cat in enumerate(self.BB_CLASS_LABELS)}
        self.bb_id2cat = {idx: cat for idx, cat in enumerate(self.BB_CLASS_LABELS)}

        self.kp_cat2id = {cat: idx for idx, cat in enumerate(self.KP_CLASS_LABELS)}
        self.kp_id2cat = {idx: cat for idx, cat in enumerate(self.KP_CLASS_LABELS)}

        self.images, self.annotations = self.load_annotations(self.gt_path)

    def __len__(self):
        return len(self.annotations)

    def _load_image(self, idx):
        info = self.images[idx]
        img_path = str(Path(self.ds_path) / self.mode / 'images' / info['filename'])
        img = cv2.imread(img_path)  # BGR

        if self.color_layout.lower() == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        assert img is not None, f'Image Not Found {img_path}'
        h0, w0 = img.shape[:2]  # orig hw
        return img, img_path, (h0, w0), idx  # img, hw_original, hw_resized

    def __getitem__(self, index):
        image, fp, shape, img_id = self._load_image(index)
        bb_data, kp_data = self.annotations[index]

        bb, bb_labels, bb_ignore, bb_ids = bb_data
        kp, kp_labels, kp_ignore, kp2bb_ids = kp_data

        if self.transforms:
            # apply albumentations transform
            transformed = self.transforms(
                image=image,
                bboxes=bb,
                bb_classes=bb_labels,
                bb_ignore=bb_ignore,
                bb_id=bb_ids,
                keypoints=kp,
                kp_classes=kp_labels,
                kp_ignore=kp_ignore,
                kp2bb_id=kp2bb_ids,
            )
            image = transformed['image']
            bb, kp = transformed['bboxes'], transformed['keypoints']
            bb_labels, kp_labels = transformed['bb_classes'], transformed['kp_classes']
            bb_ignore, kp_ignore = transformed['bb_ignore'], transformed['kp_ignore']
            bb_ids = transformed['bb_id']
            kp2bb_ids = transformed['kp2bb_id']

        bboxes = np.zeros((len(bb), 7), dtype=np.float32)  # (img_id, cat_id, weight, x1, y1, x2, y2)
        key_points = np.zeros((len(bb), 16), dtype=np.float32)  # (img_id, x1, y1, w1, ... x5, y5, w5)

        if len(bb):
            bboxes[:, 1] = np.array(bb_labels)
            bboxes[:, 2] = np.array(bb_ignore) == 0.0
            bboxes[:, 3:] = xywh2xyxy(np.array(bb))

            _id_map = {bb_ids[i]: i for i in range(len(bboxes))}

            for i, (point, ignore, box_id) in enumerate(zip(kp, kp_ignore, kp2bb_ids)):
                lbl_id = i % 5
                weight = 1.0 if is_in_image(point, image.size()[-2:]) and not ignore else 0.0
                start_ind = 1 + 3 * lbl_id
                end_ind = 1 + 3 * (lbl_id + 1)
                box_id = _id_map.get(box_id, None)

                if box_id is not None:
                    key_points[box_id, start_ind:end_ind] = [*point, weight]

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        key_points = torch.tensor(key_points, dtype=torch.float)

        return image, bboxes, key_points, img_id, fp, shape

    def _parse_ann_line(self, line):
        values = [float(x) for x in line.strip().split()]
        bbox = np.array(values[0:4], dtype=np.float32)
        kps = np.zeros((self.NK, 3), dtype=np.float32)
        ignore = False
        if self.min_size is not None:
            assert not self.test_mode
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.min_size or h < self.min_size:
                ignore = True
        if len(values) > 4:
            if len(values) > 5:
                kps = np.array(values[4:19], dtype=np.float32).reshape((self.NK, 3))
                for li in range(kps.shape[0]):
                    if (kps[li, :] == -1).all():
                        kps[li][2] = 0.0    # weight = 0, ignore
                    else:
                        assert kps[li][2] >= 0
                        kps[li][2] = 1.0    # weight
            else:
                if not ignore:
                    ignore = (values[4] == 1)

        return dict(bbox=bbox, kps=kps, ignore=ignore, cat='Face')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        name = None
        bbox_map = {}
        for line in open(ann_file, 'r'):
            line = line.strip()
            if line.startswith('#'):
                value = line[1:].strip().split()
                name = value[0]
                width = int(value[1])
                height = int(value[2])

                bbox_map[name] = dict(width=width, height=height, objs=[])
                continue

            assert name is not None
            assert name in bbox_map
            bbox_map[name]['objs'].append(line)

        data_infos = []
        for name in bbox_map:
            item = bbox_map[name]
            width = item['width']
            height = item['height']
            vals = item['objs']

            objs = []
            for line in vals:
                data = self._parse_ann_line(line)
                if data is None:
                    continue
                objs.append(data)   # data is (bbox, kps, cat)

            # if len(objs) == 0:
            #     continue

            data_infos.append(dict(filename=name, width=width, height=height, objs=objs))

        out_ann = []
        images = []

        for info in data_infos:
            objects = info['objs']
            images.append({k: info[k] for k in ['filename', 'width', 'height']})

            n_anns = len(objects)

            bb = np.zeros((n_anns, 4), dtype=np.float32)
            bb_labels = np.zeros((n_anns, 1), dtype=np.int32)
            bb_ignore = np.zeros((n_anns, 1), dtype=np.bool_)
            bb_ids = np.zeros((n_anns, 1), dtype=np.int32)

            kp = np.zeros((n_anns, self.NK, 2), dtype=np.float32)
            kp_labels = np.zeros((n_anns, self.NK), dtype=np.int32)
            kp_ignore = np.zeros((n_anns, self.NK), dtype=np.bool_)
            kp_bb_ids = np.zeros((n_anns, self.NK), dtype=np.int32)

            for idx, obj in enumerate(objects):
                bb[idx, :] = np.array(obj['bbox'], dtype=np.float32)
                bb_labels[idx, :] = self.bb_cat2id[obj['cat']]
                bb_ignore[idx, :] = False  # todo:
                bb_ids[idx, :] = idx

                kp[idx, :, :] = obj['kps'][:, :2]
                kp_labels[idx, :] = [self.kp_cat2id[_] for _ in self.KP_CLASS_LABELS]
                kp_ignore[idx, :] = obj['kps'][:, 2] == 0
                kp_bb_ids[idx, :] = [idx for _ in self.KP_CLASS_LABELS]

            bb = xyxy2xywh(bb)

            kp[kp == -1] = 0    # replace -1 with 0

            img_shape = (info['height'], info['width'])

            bb = self.validate_bb(bb, img_shape)
            kp, ignore = self.validate_kp(kp, bb)

            bb_data = [bb, bb_labels.flatten(), bb_ignore.flatten(), bb_ids.flatten()]
            kp_data = [kp, kp_labels, kp_ignore, kp_bb_ids]

            # assert len(bb_data[0]) > 0
            # kp_data['kp'] = self.validate_kp(kp_data['kp'], bb_data['bb'])

            kp_data[0] = kp_data[0].reshape(-1, 2)
            kp_data[1] = kp_data[1].flatten()
            kp_data[2] = kp_data[2].flatten()
            kp_data[3] = kp_data[3].flatten()
            out_ann.append((bb_data, kp_data))

        return images, out_ann

    def validate_bb(self, bb, img_shape):
        bb = xywh2xyxy(np.array(bb))
        h, w = img_shape
        bb[:, 0::2] = bb[:, 0::2].clip(0, w - 1)
        bb[:, 1::2] = bb[:, 1::2].clip(0, h - 1)
        bb = xyxy2xywh(bb)
        return bb

    def validate_kp(self, kp, bb):
        """
        Clip key points to the bbox size. If kp coords differ more than by 1 pixel: kp is marked as ignore
        bb: np.array(nl, 4)
        kp: np.array(nl, 5, 2)
        """
        ignore_list = []
        for i in range(len(kp)):
            box = bb[i]

            x1, y1, x2, y2 = box[0], box[1], box[0]+box[2], box[1]+box[3]

            # set outbound kps as ignored
            x_ignore_1 = kp[i, :, 0] < x1 - 1
            x_ignore_2 = kp[i, :, 0] > x2
            y_ignore_1 = kp[i, :, 1] < y1 - 1
            y_ignore_2 = kp[i, :, 1] > y2
            ignore = np.sum(np.stack([x_ignore_1, x_ignore_2, y_ignore_1, y_ignore_2]), axis=0) > 0

            # clip all key points to the bbox size
            kp[i, :, 0] = kp[i, :, 0].clip(x1, x2 - 1)
            kp[i, :, 1] = kp[i, :, 1].clip(y1, y2 - 1)

            ignore_list.append(ignore)
        return kp, np.array(ignore_list)

    @property
    def get_ds_name(self):
        return str(Path(self.ds_path).name)

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


if __name__ == '__main__':
    import albumentations as A
    import albumentations_experimental
    from albumentations.pytorch.transforms import ToTensorV2

    from detpack.datasets.transforms import RandomSmartCrop
    ds_path = '/datasets/widerface'
    transforms = A.Compose([
        RandomSmartCrop(p=1.0),
        albumentations_experimental.HorizontalFlipSymmetricKeypoints(symmetric_keypoints=[[0, 1], [2, 2], [3, 4]], p=0.5),
        A.ColorJitter(hue=0.015, saturation=0.7, contrast=0.2, brightness=0.4, p=1.0),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196]),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(format='coco', min_visibility=0.7, label_fields=['bb_classes', 'bb_ignore', 'bb_id']),
        keypoint_params=A.KeypointParams(format='xy', label_fields=['kp_classes', 'kp2bb_id', 'kp_ignore'],
                                         remove_invisible=False)
    )
    ds = WiderFaceDataset(ds_path, transforms=transforms, mode='train')
    for i in range(len(ds)):
        k = ds[i]

    k = 5