import torch
import torch.nn as nn
import torchvision
from typing import List, Tuple


from src.upd_scrfd.resnet import ResNetV1e
from src.upd_scrfd.neck import PAFPN
from src.upd_scrfd.head import SCRFDHead
from src.upd_scrfd.utils import distance2bbox, distance2kps
from src.upd_scrfd.utils import parameter_list
from src.upd_scrfd.anchor import AnchorGenerator


class SCRFD_10G(nn.Module):
    def __init__(self, nc):
        super().__init__()

        self.num_classes = nc

        block_cfg = dict(
            block='BasicBlock',
            stage_blocks=(3, 4, 2, 3),
            stage_planes=[56, 88, 88, 224]
        )
        self.backbone = ResNetV1e(depth=0, base_channels=56, num_stages=4, block_cfg=block_cfg, norm_eval=False)

        self.strides = (8, 16, 32)

        self.neck = PAFPN(
            in_channels=[56, 88, 88, 224],
            out_channels=56,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=3,
        )

        self.bbox_head = SCRFDHead(
            num_classes=self.num_classes,
            in_channels=56,
            feat_channels=80,
            stacked_convs=3,
            dw_conv=False,
            use_scale=True,
            num_anchors=(2, 2, 2),
            strides=(8, 16, 32),
        )

        self.anchor_generator = AnchorGenerator(
            aspect_ratios=(1.0, ),
            sizes=(16, 64, 256),
            anchor_strides=[8, 16, 32]
        )

    def forward(self, x):
        out = self._forward_kd_train(x)
        return out

    @torch.jit.ignore(drop=True)
    def _forward_train(self, x):
        device = x.device
        backbone_feats = self.backbone(x)
        neck_feat = self.neck(backbone_feats)
        anchors = self.anchor_generator(x, neck_feat, device)
        cls_scores, bboxes, key_points = self.bbox_head(neck_feat)
        return cls_scores, bboxes, key_points, anchors

    def _forward_kd_train(self, x):
        device = x.device
        backbone_feats = self.backbone(x)
        neck_feat = self.neck(backbone_feats)
        anchors = self.anchor_generator(x, neck_feat, device)
        cls_scores, bboxes, key_points = self.bbox_head(neck_feat)
        return neck_feat, (cls_scores, bboxes, key_points, anchors)

    def _forward(self, x):
        backbone_feats = self.backbone(x)
        neck_feat = self.neck(backbone_feats)
        cls_scores, bboxes, key_points = self.bbox_head(neck_feat)
        return cls_scores, bboxes, key_points

    def preprocess(self, x):
        return x

    @staticmethod
    def get_anchor_centers(fm_h, fm_w, stride, na):
        xv, yv = torch.meshgrid(torch.arange(fm_w), torch.arange(fm_h), indexing='ij')
        anchor_centers = torch.stack([yv, xv], dim=-1).float()
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        anchor_centers = torch.stack([anchor_centers] * na, dim=1).reshape((-1, 2))
        return anchor_centers

    def postprocess(self, raw_pred, iou_thresh, conf_thresh):
        cls_scores, bboxes, key_points, anchors = raw_pred

        bb_per_lvl = []
        kp_per_lvl = []
        scores_per_lvl = []
        labels_per_lvl = []
        per_lvl_batch_ind = []

        for stride_idx, stride in enumerate(self.strides):
            bbox = bboxes[stride_idx]
            kps = key_points[stride_idx]

            device = bbox.device

            b, AxC, h, w = bbox.shape
            na = AxC // 4
            c = 4
            bbox = bbox.view(b, na, c, h, w)
            bbox = bbox.permute(0, 3, 4, 1, 2)
            bbox = bbox.reshape(-1, c)
            bbox *= stride

            anchor_centers = self.get_anchor_centers(h, w, stride, na)
            anchor_centers = anchor_centers.repeat(b, 1)
            anchor_centers = anchor_centers.to(device)

            bbox = distance2bbox(anchor_centers, bbox)
            bb_per_lvl.append(bbox)

            b, AxC, h, w = kps.shape
            na = AxC // 10
            c = 10
            kps = kps.view(b, na, c, h, w)
            kps = kps.permute(0, 3, 4, 1, 2)
            kps = kps.reshape(-1, c)
            kps *= stride

            kps = distance2kps(anchor_centers, kps)
            kp_per_lvl.append(kps)

            scores = cls_scores[stride_idx]
            b, AxC, h, w = scores.shape
            na = AxC // self.num_classes
            c = self.num_classes
            scores = scores.view(b, na, c, h, w)
            scores = scores.permute(0, 3, 4, 1, 2)
            scores = scores.reshape(-1, c)
            scores = scores.sigmoid()
            scores, labels = torch.max(scores, dim=1)

            scores_per_lvl.append(scores)
            labels_per_lvl.append(labels)

            batch_ind = torch.arange(b, device=device).view(-1, 1).repeat(1, na * w * h).flatten()   # b x (na*w*h)
            per_lvl_batch_ind.append(batch_ind)

        bboxes = torch.cat(bb_per_lvl)
        kps = torch.cat(kp_per_lvl)
        scores = torch.cat(scores_per_lvl)
        labels = torch.cat(labels_per_lvl)
        idxs = torch.cat(per_lvl_batch_ind)

        # todo: returns tuple check?
        is_pos = torch.where(scores > conf_thresh)[0]
        bboxes = bboxes[is_pos]
        kps = kps[is_pos]
        scores = scores[is_pos]
        labels = labels[is_pos]
        idxs = idxs[is_pos]

        keep_after_nms = torchvision.ops.batched_nms(bboxes, scores, idxs, iou_threshold=iou_thresh)

        bboxes = bboxes[keep_after_nms]
        kps = kps[keep_after_nms]
        scores = scores[keep_after_nms]
        labels = labels[keep_after_nms]
        idxs = idxs[keep_after_nms]

        out_bboxes = [[] for _ in range(b)]
        out_scores = [[] for _ in range(b)]
        out_labels = [[] for _ in range(b)]
        out_key_points = [[] for _ in range(b)]
        for bbox, kp, score, lbl, idx in zip(bboxes, kps, scores, labels, idxs):
            out_scores[idx].append(score)
            out_bboxes[idx].append(bbox)
            out_key_points[idx].append(kp)
            out_labels[idx].append(lbl)
        # cls_score, labels, bbox_pred, kps
        return out_scores, out_labels, out_bboxes, out_key_points

    def predict(self, x, iou_thresh, conf_thresh):
        x = self.preprocess(x)
        out = self._forward(x)
        out = self.postprocess(out, iou_thresh, conf_thresh)
        return out

    def load_from_checkpoint(self, ckpt_fp, strict=True, verbose=False):
        ckpt = torch.load(ckpt_fp)

        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            # from collections import OrderedDict
            #
            # tmp_sd = OrderedDict()
            # for k, v in sd.items():
            #     if k.startswith('backbone'):
            #         tmp_sd[k.replace('backbone.', '')] = v
            # self.backbone.load_state_dict(tmp_sd)
            #
            # tmp_sd = OrderedDict()
            # for k, v in sd.items():
            #     if k.startswith('neck'):
            #         k = k.replace('.conv.', '.')
            #         tmp_sd[k.replace('neck.', '')] = v
            # self.neck.load_state_dict(tmp_sd)
            #
            # tmp_sd = OrderedDict()
            # sd1 = {k: v for k, v in self.state_dict().items() if k.startswith('bbox_head.stride_heads.0') and 'scale.scale' not in k}
            # sd2 = {k: v for k, v in sd.items() if '(8, 8)' in k}
            # for (k_1, v_1), (k_2, v_2) in zip(sd1.items(), sd2.items()):
            #     tmp_sd[k_1.replace('bbox_head.stride_heads.0.', '')] = v_2
            # tmp_sd['scale.scale'] = sd['bbox_head.scales.0.scale']
            # self.bbox_head.stride_heads[0].load_state_dict(tmp_sd)
            #
            # tmp_sd = OrderedDict()
            # sd1 = {k: v for k, v in self.state_dict().items() if k.startswith('bbox_head.stride_heads.1') and 'scale.scale' not in k}
            # sd2 = {k: v for k, v in sd.items() if '(16, 16)' in k}
            # for (k_1, v_1), (k_2, v_2) in zip(sd1.items(), sd2.items()):
            #     tmp_sd[k_1.replace('bbox_head.stride_heads.1.', '')] = v_2
            # tmp_sd['scale.scale'] = sd['bbox_head.scales.1.scale']
            # self.bbox_head.stride_heads[1].load_state_dict(tmp_sd)
            #
            # tmp_sd = OrderedDict()
            # sd1 = {k: v for k, v in self.state_dict().items() if k.startswith('bbox_head.stride_heads.2') and 'scale.scale' not in k}
            # sd2 = {k: v for k, v in sd.items() if '(32, 32)' in k}
            # for (k_1, v_1), (k_2, v_2) in zip(sd1.items(), sd2.items()):
            #     tmp_sd[k_1.replace('bbox_head.stride_heads.2.', '')] = v_2
            # tmp_sd['scale.scale'] = sd['bbox_head.scales.2.scale']
            # self.bbox_head.stride_heads[2].load_state_dict(tmp_sd)
            #
            # sd = {'state_dict': self.state_dict()}
            # torch.save(sd, '../weights/upd_SCRFD_10G_KPS.pth')
            # k = 5

            self.load_state_dict(state_dict=ckpt['state_dict'], strict=strict)
        else:
            self.load_state_dict(state_dict=ckpt['model'], strict=strict)
        del ckpt

    def get_param_groups(self, no_decay_bn_filter_bias, wd):
        return parameter_list(self.named_parameters, weight_decay=wd, no_decay_bn_filter_bias=no_decay_bn_filter_bias)


if __name__ == '__main__':
    device = 'cpu'
    from time import time

    data = torch.rand((1, 3, 320, 320)).to(device)
    model = SCRFD_10G(nc=1).eval().to(device)

    time_avg = []
    with torch.no_grad():
        for i in range(200):
            t0 = time()
            _ = model(data)
            t1 = time()
            if i > 20:
                time_avg.append(t1 - t0)

    print(f'Avg inference speed ({device}): {1000 * sum(time_avg) / len(time_avg)}ms')
    k = 5
