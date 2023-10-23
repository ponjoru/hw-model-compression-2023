import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader

from src.upd_scrfd.scrfd import SCRFD_500M
from src.upd_scrfd.head import SCRFDHead
from src.upd_scrfd.neck import PAFPN
from src.dataset import WiderFaceDataset
from src.utils import get_transforms, compute_pytorch_latency
from src.upd_scrfd.utils import postprocess
from homeworks.hw1_validation import eval


class QPAFPN(PAFPN):
    def __init__(self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False
    ):
        super(QPAFPN, self).__init__(in_channels, out_channels, num_outs, start_level, end_level, add_extra_convs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            x = lateral_conv(inputs[i + self.start_level])
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            a = laterals[i - 1]
            b = F.interpolate(laterals[i], size=prev_shape, mode='nearest')
            laterals[i - 1] = self.skip_add.add(a, b)

        # build outputs
        # part 1: from original levels
        inter_outs = []
        for i, layer in enumerate(self.fpn_convs):
            x = layer(laterals[i])
            inter_outs.append(x)

        # part 2: add bottom-up path
        for i, layer in enumerate(self.downsample_convs):
            inter_outs[i + 1] = self.skip_add.add(inter_outs[i + 1], layer(inter_outs[i]))

        outs = []
        outs.append(inter_outs[0])
        for i, layer in enumerate(self.pafpn_convs):
            outs.append(layer(inter_outs[i+1]))

        return outs


class QSCRFD_500M(SCRFD_500M):
    def __init__(self, nc):
        super(QSCRFD_500M, self).__init__(nc)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.neck = QPAFPN(
            in_channels=[40, 72, 152, 288],
            out_channels=16,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=3,
        )

        self.bbox_head = SCRFDHead(
            num_classes=self.num_classes,
            in_channels=16,
            stacked_convs=2,
            dw_conv=True,
            num_anchors=(2, 2, 2),
            strides=(8, 16, 32),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self._forward(x)
        return x

    def _forward(self, x: torch.Tensor):
        backbone_feats = self.backbone(x)

        neck_feat = self.neck(backbone_feats)

        cls_scores, bboxes, key_points = self.bbox_head(neck_feat)

        cls_scores = [self.dequant(s) for s in cls_scores]
        bboxes = [self.dequant(bb) for bb in bboxes]
        key_points = [self.dequant(kp) for kp in key_points]

        return cls_scores, bboxes, key_points


def calibrate(model, dataloader, neval_batches=None, device='cpu'):
    model.eval()
    model.to(device)

    if neval_batches:
        print(f'Limiting number of batches to: {neval_batches}. Available: {len(dataloader)}')

    n = neval_batches if neval_batches else len(dataloader)

    desc = f'Calibrating'
    bar_format = '{l_bar}{bar}{r_bar}'
    pbar = tqdm(dataloader, bar_format=bar_format, total=n, desc=desc)
    with torch.no_grad():
        for batch_i, batch in enumerate(pbar):
            if batch_i > (n-1):
                break

            img = batch['image']
            img = img.to(device)

            _ = model(img)


def ptq(fp32_model, dataloader, save_path, optimize_for_mobile=False):
    stack = []
    pairs = []
    for i, (k, v) in enumerate(fp32_model.named_modules()):
        if isinstance(v, nn.Conv2d):
            stack.append((k, v))
            continue
        if isinstance(v, nn.BatchNorm2d):
            if len(stack):
                conv_k, conv_v = stack.pop()
                pairs.append([conv_k, k])
                stack = []
            continue

    print('Fusing conv bn layers')
    torch.quantization.fuse_modules(fp32_model, modules_to_fuse=pairs, inplace=True)

    fp32_model.qconfig = torch.quantization.get_default_qconfig(backend='qnnpack')
    print(f'Quantization config: {fp32_model.qconfig}')

    fp32_model = torch.quantization.prepare(fp32_model)

    # calibration
    print(' Stating calibration '.center(100, '='))
    calibrate(fp32_model, dataloader, neval_batches=None, device='cpu')

    print('Converting model '.center(100, '='))
    int8_model = torch.quantization.convert(fp32_model)

    print('Scripting model '.center(100, '='))
    int8_model = torch.jit.script(int8_model)

    if optimize_for_mobile:
        print('Optimizing for mobile model '.center(100, '='))
        int8_model = torch.utils.mobile_optimizer.optimize_for_mobile(int8_model, backend='CPU')

    print('Saving model '.center(100, '='))
    torch.jit.save(int8_model, save_path)


if __name__ == '__main__':
    device = 'cpu'
    ds_path = '../data/widerface'
    print('SCRFD_500M_KPS model')
    # # quantize
    # model = QSCRFD_500M(nc=1).eval().to(device)
    # model.load_from_checkpoint('../weights/SCRFD_500M_KPS.pth')
    # dataset = WiderFaceDataset(ds_path, 'train', min_size=None, transforms=get_transforms('val'), color_layout='RGB')
    # dataloader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn)
    # ptq(model, dataloader, '../weights/scrfd_500m_kps_int8.pth', optimize_for_mobile=False)

    # eval models
    dataset = WiderFaceDataset(ds_path, 'val', min_size=None, transforms=get_transforms('val'), color_layout='RGB')
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn)
    postproc = lambda x: postprocess(x, conf_thresh=0.02, iou_thresh=0.45)

    # ------------------------- int8 -------------------------
    print('int8 model'.center(100, '='))
    int8_model = torch.jit.load('../weights/scrfd_500m_kps_int8.pth')
    int8_latency = compute_pytorch_latency(int8_model, (1, 3, 640, 640))
    print(f'latency: {int8_latency:.2f}ms')
    int8_metrics = eval(int8_model, dataloader, postproc, device)
    print(int8_metrics)

    # ------------------------- fp32 -------------------------
    print('fp32 model'.center(100, '='))
    fp32_model = SCRFD_500M(nc=1).eval().to(device)
    fp32_model.load_from_checkpoint('../weights/upd_SCRFD_500M_KPS.pth')
    fp32_latency = compute_pytorch_latency(fp32_model, (1, 3, 640, 640))
    print(f'latency: {fp32_latency:.2f}ms')
    fp32_metrics = eval(fp32_model, dataloader, postproc, device)
    print(fp32_metrics)

    """
    SCRFD_500M_KPS model
    =============================================int8 model=============================================
    latency: 20.50ms
    Evaluating: 100%|██████████| 404/404 [03:01<00:00,  2.23it/s]
    Detection Metrics:
    easy_AP: 0.9023, medium_AP: 0.8731, hard_AP: 0.6584
    =============================================fp32 model=============================================
    latency: 49.20ms
    Evaluating: 100%|██████████| 404/404 [05:22<00:00,  1.25it/s]
    Detection Metrics:
    easy_AP: 0.9071, medium_AP: 0.8805, hard_AP: 0.6768
    """