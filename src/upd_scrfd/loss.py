import math
import torch
import numpy as np
from detpack.modules.atss import ATSSModule
from detpack.torch_utils import bbox_overlaps, distance2bbox, kps2distance


class MultiOutputDetectionLoss(object):
    NK = 5

    def __init__(self, cls_loss, bb_loss, kp_loss):
        self.cls_loss = cls_loss
        self.bb_reg_loss = bb_loss
        self.kp_reg_loss = kp_loss

        self.loss_kps_std = 1.0

        self.strides = [8, 16, 32]
        self.cls_out_channels = 1
        self.num_classes = 1
        self.atss = ATSSModule(top_k=9)

        self.use_qscore = True

    @property
    def num_loss_items(self):
        return 4

    @property
    def loss_items_names(self):
        cls_name = 'cls_' + self.cls_loss.name
        bb_name = 'bb_' + self.bb_reg_loss.name
        kp_name = 'kp_' + self.kp_reg_loss.name
        return [cls_name, bb_name, kp_name, 'Total']

    def __call__(self, prediction, targets, img_shape):
        box_cls, bbox_pred, kp_pred, anchors = prediction

        per_lvl_targets, per_lvl_weights = self.prepare_targets(targets, anchors)

        per_lvl_predictions = self.prepare_predictions(prediction)

        per_lvl_anchors = self.prepare_anchors(anchors)

        num_total_samples = sum([((t >= 0) & (t < self.num_classes)).sum() for t in per_lvl_targets[0]])

        losses = []
        wts = []
        for i, stride in enumerate(self.strides):
            a = per_lvl_anchors[i]
            pred = [item[i].float() for item in per_lvl_predictions]
            gt = [item[i] for item in per_lvl_targets]
            weights = [item[i] for item in per_lvl_weights]

            loss, wt = self.compute_losses(a, stride, pred, gt, weights, num_total_samples)
            losses.append(loss)
            wts.append(wt)

        loss, loss_items = self.aggregate_losses(losses, avg_factor=max(sum(wts), 1.0))

        return loss, loss_items

    def compute_losses(self, anchors, stride, predictions, targets, weights, num_total_samples):
        cls_score, bbox_pred, kps_pred = predictions
        labels, bbox_targets, kps_targets = targets

        device = cls_score.device

        num_total_samples = max(num_total_samples, 1.0)

        anchors = anchors.reshape(-1, 4)

        # reshape label preds & gts
        cls_score = cls_score.reshape(-1, self.cls_out_channels)
        labels = labels.reshape(-1)

        # reshape bbox preds & gts
        bbox_pred = bbox_pred.reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)

        # reshape kp preds & gts
        kps_pred = kps_pred.reshape(-1, self.NK * 2)
        kps_targets = kps_targets.reshape((-1, self.NK * 2))

        cls_weights, gt_bb_weights, gt_kp_weights = [w.reshape(-1) for w in weights]

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        score = gt_bb_weights.new_zeros(labels.shape, dtype=torch.float32)

        # if there are positive detections found - compute regression losses
        if len(pos_inds) > 0:
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride
            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            # compute bb reg loss
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bb_weights = gt_bb_weights[pos_inds] * weight_targets
            pos_decode_bbox_targets = pos_bbox_targets / stride
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred)
            loss_bbox = self.bb_reg_loss(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bb_weights,
                avg_factor=1.0)

            # compute kp reg loss
            pos_kps_targets = kps_targets[pos_inds]
            pos_kps_pred = kps_pred[pos_inds]
            pos_kps_weights = gt_kp_weights[pos_inds] * weight_targets
            pos_kps_weights = pos_kps_weights.reshape((-1, 1))
            pos_kps_targets = pos_kps_targets / stride
            pos_decode_kps_targets = kps2distance(pos_anchor_centers, pos_kps_targets)
            pos_decode_kps_pred = pos_kps_pred
            loss_kps = self.kp_reg_loss(
                pos_decode_kps_pred * self.loss_kps_std,
                pos_decode_kps_targets * self.loss_kps_std,
                weight=pos_kps_weights,
                avg_factor=1.0)

            # compute qscore if needed
            if self.use_qscore:
                score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
            else:
                score[pos_inds] = 1.0
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_kps = kps_pred.sum() * 0
            weight_targets = torch.tensor(0).to(device)

        loss_cls = self.cls_loss(
            cls_score, (labels, score),
            weight=cls_weights,
            avg_factor=num_total_samples)

        return (loss_cls, loss_bbox, loss_kps), weight_targets.sum()

    def prepare_targets(self, targets, anchors):
        """ Assign anchors to targets and reshape targets and weights to per stride format """
        gt_cls, gt_bb, gt_kp = [], [], []
        gt_cls_weights, gt_bb_weights, gt_kp_weights = [], [], []

        bb, kp = targets['bb'], targets['kp']
        device = bb.device

        batch_size = len(anchors)
        for im_i in range(batch_size):
            boxes = bb[bb[:, 0] == im_i]
            kps = kp[kp[:, 0] == im_i]
            anchors_per_im = torch.cat([a.bbox for a in anchors[im_i]])

            # parse ground truth targets
            cls_per_im = boxes[:, 1].unsqueeze(1)
            bb_per_im = boxes[:, 3:]
            kp_per_im = kps[:, 1:].reshape(-1, 5, 3)[:, :, :-1]
            cls_weights_per_im = boxes[:, 2].unsqueeze(1)
            bb_weights_per_im = boxes[:, 2].unsqueeze(1)
            kp_weights_per_im = kps[:, 1:].reshape(-1, 5, 3)[:, :, -1]

            if len(cls_per_im) == 0:
                gt_cls.append(torch.full((len(anchors_per_im), 1), self.num_classes, dtype=torch.long, device=device))
                gt_bb.append(torch.zeros((len(anchors_per_im), 4), dtype=torch.float, device=device))
                gt_kp.append(torch.zeros((len(anchors_per_im), 5, 2), dtype=torch.float, device=device))

                gt_cls_weights.append(torch.ones((len(anchors_per_im), 1), dtype=torch.long, device=device))
                gt_bb_weights.append(torch.zeros((len(anchors_per_im), 1), dtype=torch.float, device=device))
                gt_kp_weights.append(torch.zeros((len(anchors_per_im), 5), dtype=torch.float, device=device))
                continue

            # match indices
            num_anchors_per_level = [len(anchors_per_level) for anchors_per_level in anchors[im_i]]
            anchors_to_gt_indices, anchors_to_gt_values = self.atss(anchors_per_im, bb_per_im, num_anchors_per_level)

            # select & add class labels
            cls_labels_per_im = cls_per_im[anchors_to_gt_indices]
            cls_labels_per_im[anchors_to_gt_values == -math.inf] = self.num_classes
            gt_cls.append(cls_labels_per_im)

            # select & add bbox targets
            gt_bb.append(bb_per_im[anchors_to_gt_indices])

            # select & add keypoint targets
            gt_kp.append(kp_per_im[anchors_to_gt_indices])

            # select & add targets weights
            gt_cls_weights.append(cls_weights_per_im[anchors_to_gt_indices])
            gt_bb_weights.append(bb_weights_per_im[anchors_to_gt_indices])
            gt_kp_weights.append(kp_weights_per_im[anchors_to_gt_indices])

        assigned_targets = (gt_cls, gt_bb, gt_kp)
        assigned_weights = (gt_cls_weights, gt_bb_weights, gt_kp_weights)

        # split targets and weights by levels
        start_idx = 0
        num_anchors_per_stride = [len(a) for a in anchors[0]]
        per_level_targets = [[], [], []]
        per_level_weights = [[], [], []]
        for na in num_anchors_per_stride:
            end_idx = start_idx + na

            for i, target_item in enumerate(assigned_targets):
                per_level_targets[i].append(torch.stack(target_item)[:, start_idx:end_idx, :])

            for i, weight_item in enumerate(assigned_weights):
                per_level_weights[i].append(torch.stack(weight_item)[:, start_idx:end_idx, :])

            start_idx = end_idx

        return per_level_targets, per_level_weights

    def prepare_predictions(self, predictions):
        """ Reshape predictions to per stride format """
        cls_pred, bb_pred, kp_pred, _ = predictions

        per_lvl_cls_pred, per_lvl_bb_pred, per_lvl_kp_pred = [], [], []

        for p in cls_pred:
            N, AxC, H, W = p.shape
            A = AxC // 1
            C = 1
            p = p.view(N, A, C, H, W)
            p = p.permute(0, 3, 4, 1, 2)
            per_lvl_cls_pred.append(p.reshape(N, -1, C))

        for p in bb_pred:
            N, AxC, H, W = p.shape
            A = AxC // 4
            C = 4
            p = p.view(N, A, C, H, W)
            p = p.permute(0, 3, 4, 1, 2)
            per_lvl_bb_pred.append(p.reshape(N, -1, C))

        for p in kp_pred:
            N, AxC, H, W = p.shape
            A = AxC // (self.NK * 2)
            C = (self.NK * 2)
            p = p.view(N, A, C, H, W)
            p = p.permute(0, 3, 4, 1, 2)
            per_lvl_kp_pred.append(p.reshape(N, -1, C))

        predictions = [per_lvl_cls_pred, per_lvl_bb_pred, per_lvl_kp_pred]
        return predictions

    def prepare_anchors(self, anchors):
        """ Reshape anchors to per stride format """
        per_lvl_anchors = [[] for _ in range(len(anchors[0]))]
        for anchors_per_image in anchors:
            for i, a in enumerate(anchors_per_image):
                per_lvl_anchors[i].append(a.bbox)
        per_lvl_anchors = [torch.stack(a, dim=0) for a in per_lvl_anchors]
        return per_lvl_anchors

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def aggregate_losses(self, losses, avg_factor):
        losses = torch.stack([torch.stack(l) for l in losses])
        losses[:, 1:] /= avg_factor

        loss_items = losses.mean(dim=0)
        loss = loss_items.mean()
        loss_items = torch.cat([loss_items.detach(), loss.detach().unsqueeze(0)])

        return loss, loss_items

