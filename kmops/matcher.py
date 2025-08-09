import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from einops import rearrange


# oks based loss
def oks_and_kpts_loss(src_kpts, tgt_kpts, area, sigmas):
    d = (src_kpts[:, None, :, 0] - tgt_kpts[None, :, :, 0]) ** 2 + \
        (src_kpts[:, None, :, 1] - tgt_kpts[None, :, :, 1]) ** 2
    d = d / (area[:, None] * 4 * sigmas[None, :] ** 2 + 1e-9)
    kpts_mask = tgt_kpts[:, :, 2]
    d = torch.exp(-d) * kpts_mask
    d = d.sum(dim=-1) / (kpts_mask.sum(dim=-1) + 1e-9)
    cost_oks = 1 - d

    cost_kpts = torch.abs(
        src_kpts[:, None, :, :] - tgt_kpts[None, :, :, :2])
    cost_kpts = cost_kpts * kpts_mask[None, :, :, None]
    cost_kpts = cost_kpts.sum(dim=[-1, -2])
    return cost_oks, cost_kpts


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general, there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, coef_class, coef_kpts, coef_oks, coef_conf,
                 num_keypoints):
        """Creates the matcher

        Params:
            coef_class: This is the relative weight of the classification
                        error in the matching cost
            coef_kpts: This is the relative weight of the keypoints error
                       in the matching cost
            coef_oks: This is the relative weight of the Object Keypoint
                      Similarity (OKS) error in the matching cost
            num_keypoints: The number of keypoints to estimate
        """
        super().__init__()
        self.coef_class = coef_class
        self.coef_kpts = coef_kpts
        self.coef_oks = coef_oks
        self.coef_conf = coef_conf
        assert coef_class != 0 or coef_kpts != 0 or coef_oks != 0, \
            "all costs cant be 0"

        self.num_keypoints = num_keypoints

        if num_keypoints == 17:
            sigmas = torch.tensor([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=torch.float32) / 10.0
        else:
            sigmas = torch.ones(num_keypoints, dtype=torch.float32) / 10.0

        self.register_buffer('sigmas', sigmas, persistent=False)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching in order

        Params:
            outputs: This is a dict that contains at least these entries:
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

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
            - index_i is the indices of the selected predictions
            - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We rearrange to compute the cost matrices in a batch
        out_prob = rearrange(
            outputs["pred_logits"], 'b q c -> (b q) c').sigmoid()
        out_kpts = rearrange(outputs["pred_kpts"], 'b q k n -> (b q) k n')
        out_disp = rearrange(outputs["pred_disp"], 'b q k n -> (b q) k n')
        out_kpts_conf = rearrange(
            outputs["pred_conf"], 'b q k n -> (b q) k n').sigmoid()

        # Also concat the target labels and boxes
        tgt_ids = torch.cat(targets["labels"]).long()
        tgt_area = torch.cat(targets["area"])
        tgt_kpts = torch.cat(targets["kpts"])
        tgt_disp = torch.cat(targets["disp"])
        tgt_kpts_vis = tgt_kpts[:, :, 2:]

        # Compute the classification cost
        alpha = 0.25
        gamma = 2
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * \
            (-torch.log(1 - out_prob + 1e-8))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * \
            (-torch.log(out_prob + 1e-8))
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_oks, cost_kpts = \
            oks_and_kpts_loss(out_kpts, tgt_kpts, tgt_area, self.sigmas)

        cost_disp = torch.abs(out_disp[:, None] - tgt_disp[None, :])
        cost_disp = cost_disp * tgt_kpts_vis[None]
        cost_disp = cost_disp.sum(dim=[-1, -2])

        cost_kpts = cost_kpts + cost_disp * 2

        cost_conf = torch.abs(out_kpts_conf[:, None] - tgt_kpts_vis[None, :])
        cost_conf = cost_conf.sum(dim=[-1, -2])

        # Final cost matrix
        C = self.coef_class * cost_class \
            + self.coef_oks * cost_oks \
            + self.coef_kpts * cost_kpts \
            + self.coef_conf * cost_conf
        C = rearrange(C, '(b q) t -> b q t', b=bs, q=num_queries).cpu()

        sizes = [len(v) for v in targets['labels']]
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
