import torch
from torch import nn
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the
                 binary classification label for each element in inputs
                 (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


def oks_and_kpts_loss(src_kpts, tgt_kpts, area, sigmas):
    d = (src_kpts[..., 0] - tgt_kpts[..., 0]) ** 2 + \
        (src_kpts[..., 1] - tgt_kpts[..., 1]) ** 2
    d = d / (area[:, None] * 4 * sigmas[None, :] ** 2 + 1e-9)
    kpts_mask = tgt_kpts[:, :, 2]
    d = torch.exp(-d) * kpts_mask
    d = d.sum(dim=-1) / (kpts_mask.sum(dim=-1) + 1e-9)
    loss_oks = 1 - d

    loss_kpts = F.l1_loss(
        src_kpts, tgt_kpts[..., :2], reduction='none')
    loss_kpts = loss_kpts * tgt_kpts[:, :, 2:]
    loss_kpts = loss_kpts.sum(dim=[-1, -2])
    return loss_oks, loss_kpts


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the
           outputs of the model
        2) we supervise each pair of matched ground-truth / prediction
           (supervise class and box)
    """
    def __init__(self, num_classes, num_keypoints, num_queries, weight_dict,
                 matcher, matcher_o2m=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special
                         no-object category
            matcher: module able to compute a matching between targets and
                     proposals
            weight_dict: dict containing as key the names of the losses and as
                         values their relative weight.
            eos_coef: relative classification weight applied to the no-object
                      category
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.num_queries = num_queries
        self.weight_dict = weight_dict
        self.matcher = matcher
        self.matcher_o2m = matcher_o2m

        if num_keypoints == 17:
            sigmas = torch.tensor([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=torch.float32) / 10.0
        else:
            sigmas = torch.ones(num_keypoints, dtype=torch.float32) / 10.0

        self.register_buffer('sigmas', sigmas, persistent=False)

    def compute_loss(self, outputs, targets, indices):
        """
        Compute the loss for class labels (NLL), the lthe L1 regression loss,
        and the GIoU loss for bboxes

        Targets dicts must contain the key "labels" containing a tensor of dim
        [number_of_target_boxes]
        Targets dicts must contain the key "boxes" containing a tensor of dim
        [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h),
        normalized by the image size.
        """
        # retrieve the matching between the outputs of the last layer and the
        batch_idx, src_idx, _ = self._get_permutation_idx(indices)

        # Compute the classification loss
        # breakpoint()
        src_logits = outputs['pred_logits']
        tgt_labels = targets["labels"]

        # generate target for no_object classes (idx = num_classes)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device)

        # retrieve the target class according to the matching indices
        target_classes_o = torch.cat(
            [t[i] for t, (_, i) in zip(tgt_labels, indices)])

        target_classes[batch_idx, src_idx] = target_classes_o.long()

        # generate one hot encoding for target classes
        # This is faster than:
        # target_classes_onehot = F.one_hot(
        #    target_classes, num_classes=self.num_classes + 1).float()
        target_classes_onehot = torch.zeros(
            *target_classes.shape, self.num_classes + 1,
            device=target_classes.device
        )
        target_classes_onehot.scatter_(
            dim=-1,
            index=target_classes.unsqueeze(-1),
            value=1
        )

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, alpha=0.25)

        # get the area of the projection of the keypoints
        tgt_area = targets["area"]
        target_area_o = torch.cat(
            [t[i] for t, (_, i) in zip(tgt_area, indices)])

        num_boxes = target_classes_o.size(0)
        if num_boxes == 0:
            return {
                'loss_ce': loss_ce.sum(),
                'loss_oks': torch.tensor(0.0, device=src_logits.device),
                'loss_kpts': torch.tensor(0.0, device=src_logits.device)
            }

        # Compute the keypoint oks loss
        src_kpts = outputs["pred_kpts"]
        src_disp = outputs["pred_disp"]
        tgt_kpts = targets["kpts"]
        tgt_disp = targets["disp"]

        # retrieve the target keypoints according to the matching indices
        target_kpts_o = torch.cat(
            [t[i] for t, (_, i) in zip(tgt_kpts, indices)])
        target_disp_o = torch.cat(
            [t[i] for t, (_, i) in zip(tgt_disp, indices)])

        src_kpts = src_kpts[batch_idx, src_idx]
        src_disp = src_disp[batch_idx, src_idx]

        loss_oks, loss_kpts = oks_and_kpts_loss(
            src_kpts, target_kpts_o, target_area_o, self.sigmas)

        target_kpts_vis_o = target_kpts_o[..., 2:]
        loss_disp = F.l1_loss(src_disp, target_disp_o, reduction='none')
        loss_disp = loss_disp * target_kpts_vis_o
        loss_disp = loss_disp.sum(dim=[-1, -2])

        loss_kpts = loss_kpts + loss_disp * 2

        # visibility loss
        src_conf = outputs['pred_conf'][batch_idx, src_idx]
        loss_conf = sigmoid_focal_loss(src_conf, target_kpts_vis_o, alpha=0.25)

        return {'loss_ce': loss_ce.sum() / num_boxes,
                'loss_oks': loss_oks.sum() / num_boxes,
                'loss_kpts': loss_kpts.sum() / num_boxes,
                'loss_conf': loss_conf.sum() / num_boxes}

    def _get_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx, src_idx, tgt_idx = [], [], []
        for i, (src, tgt) in enumerate(indices):
            batch_idx.append(torch.full_like(src, i))
            src_idx.append(src)
            tgt_idx.append(tgt)

        return torch.cat(batch_idx), torch.cat(src_idx), torch.cat(tgt_idx)

    def indices_merge(self, num_queries, o2o_indices, o2m_indices):
        bs = len(o2o_indices)
        device = o2o_indices[0][0].device
        temp_indices = torch.zeros(
            bs, num_queries, dtype=torch.int64, device=device) - 1
        new_one2many_indices = []

        for i in range(bs):
            one2many_fg_inds = o2m_indices[i][0]
            one2many_gt_inds = o2m_indices[i][1]
            one2one_fg_inds = o2o_indices[i][0]
            one2one_gt_inds = o2o_indices[i][1]
            temp_indices[i][one2one_fg_inds] = one2one_gt_inds
            temp_indices[i][one2many_fg_inds] = one2many_gt_inds
            fg_inds = torch.nonzero(temp_indices[i] >= 0).squeeze(1)
            gt_inds = temp_indices[i][fg_inds]
            new_one2many_indices.append((fg_inds, gt_inds))

        return new_one2many_indices

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
            outputs: dict of tensors:
              "pred_logits" [batch_size, num_queries, num_classes]:
                Tensor with the classification logits
              "pred_boxes" [batch_size, num_queries, 4]:
                Tensor with the predicted box coordinates
            targets: This is a dict that contains a list of targets:
              "labels" [num_target_boxes]:
                Tensor that contains the class labels (where num_target_boxes
                is the number of ground-truth objects in the target)
              "boxes" [num_target_boxes, 4]:
                Tensor that contains the target box coordinates
        """
        losses_all = {}
        total_loss = 0
        o2o_indices_list = []

        outputs_without_aux = {k: v for k, v in outputs.items()
                               if 'enc' not in k
                               and 'aux' not in k
                               and 'o2m' not in k}
        # Retrieve the matching between the outputs of the last layer and the
        # targets
        indices = self.matcher(outputs_without_aux, targets)
        o2o_indices_list.append(indices)
        # Compute losses
        losses = self.compute_loss(outputs_without_aux, targets, indices)

        loss = 0
        for k in losses.keys():
            if k in self.weight_dict:
                loss += losses[k] * self.weight_dict[k]
            losses_all.update({f'{k}': losses[k].detach()})
        losses_all.update({'loss': loss.detach()})
        total_loss += loss

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                o2o_indices_list.append(indices)
                losses = self.compute_loss(aux_outputs, targets, indices)

                aux_loss = 0
                for k in losses.keys():
                    if k in self.weight_dict:
                        aux_loss += losses[k] * self.weight_dict[k]
                    losses_all.update({f'aux_{k}_{i}': losses[k].detach()})
                losses_all.update({f'aux_loss_{i}': aux_loss.detach()})
                total_loss += aux_loss

        if 'o2m_outputs' in outputs:
            for i, o2m_outputs in enumerate(outputs['o2m_outputs']):
                indices = self.matcher_o2m(o2m_outputs, targets)
                o2o_indices = o2o_indices_list[i]
                indices = self.indices_merge(
                    self.num_queries, o2o_indices, indices)
                losses = self.compute_loss(o2m_outputs, targets, indices)

                o2m_loss = 0
                for k in losses.keys():
                    if k in self.weight_dict:
                        o2m_loss += losses[k] * self.weight_dict[k]
                    losses_all.update({f'o2m_{k}_{i}': losses[k].detach()})
                losses_all.update({f'o2m_loss_{i}': o2m_loss.detach()})
                total_loss += o2m_loss

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets)
            losses = self.compute_loss(enc_outputs, targets, indices)

            enc_loss = 0
            for k in losses.keys():
                if k in self.weight_dict:
                    enc_loss += losses[k] * self.weight_dict[k]
                losses_all.update({f'enc_{k}': losses[k].detach()})
            losses_all.update({'enc_loss': enc_loss.detach()})
            total_loss += enc_loss

        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)

            for i, dn_outputs in enumerate(outputs['dn_aux_outputs']):
                losses = self.compute_loss(dn_outputs, targets, indices)

                dn_loss = 0
                for k in losses.keys():
                    if k in self.weight_dict:
                        dn_loss += losses[k] * self.weight_dict[k]
                    losses_all.update({f'dn_{k}_{i}': losses[k].detach()})
                losses_all.update({f'dn_loss_{i}': dn_loss.detach()})
                total_loss += dn_loss

        losses_all.update({'total_loss': total_loss})

        return losses_all

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta["dn_positive_idx"]
        dn_num_group = dn_meta["dn_num_group"]
        num_gts = [len(t) for t in targets['labels']]
        device = targets['labels'][0].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (torch.zeros(0, dtype=torch.int64, device=device),
                     torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices
