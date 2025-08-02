import torch
import torch.nn as nn
from einops import rearrange

from utils.util import kpts_to_boxes, kpts_pose_disp_to_left_right


class PostProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, target_size):
        """Scaled the predicted boxes in [0, 1] to the target size and format

        Parameters:
            outputs: raw outputs of the model
                "pred_logits" [batch_size, num_queries, num_classes]:
                Tensor with the classification logits
                "pred_poses" [batch_size, num_queries, num_keypoints, 2]:
                Tensor with the predicted keypoint coordinates
        """
        out_logits = outputs['pred_logits'].detach()
        out_pose = outputs['pred_pose'].detach()
        out_disp = outputs['pred_disp'].detach()
        (sampling_loc_l, atten_weight_l,
         sampling_loc_r, atten_weight_r) = outputs['log_infos']

        out_kpts_l, out_kpts_r \
            = kpts_pose_disp_to_left_right(out_pose, out_disp)

        assert len(target_size) == len(out_logits), \
            "target_size should have the same length as outputs"
        assert len(target_size[0]) == 2, \
            "target_size should be a tuple (height, width)"
        target_size = torch.as_tensor(target_size, device=out_logits.device)
        target_size = target_size[:, [1, 0]]  # convert to (width, height)

        preds_l, pred_r, pred_kpts_l, pred_kpts_r = [], [], [], []
        topk = {
            "box_l": [],
            "box_r": [],
            "kpts_l": [],
            "kpts_r": [],
            "loc_l": [],
            "loc_r": [],
            "weight_l": [],
            "weight_r": [],
        }

        # convert realtive [0, 1] to absolute coordinates
        out_kpts_l = out_kpts_l * rearrange(target_size, 'b n -> b 1 1 n')
        out_kpts_r = out_kpts_r * rearrange(target_size, 'b n -> b 1 1 n')
        sampling_loc_l *= rearrange(target_size, 'b n -> b 1 1 n')
        sampling_loc_r *= rearrange(target_size, 'b n -> b 1 1 n')

        boxes_l = kpts_to_boxes(out_kpts_l)
        boxes_r = kpts_to_boxes(out_kpts_r)

        for batch_idx in range(out_logits.size(0)):
            prob = out_logits[batch_idx].sigmoid()
            scores, labels = prob.max(-1, keepdim=True)

            _, indices = torch.topk(
                scores.flatten(), scores.size(0), largest=True, sorted=True)

            p_box_l = boxes_l[batch_idx, indices].detach().cpu()
            p_box_r = boxes_r[batch_idx, indices].detach().cpu()
            p_label = labels[indices].detach().cpu()
            p_score = scores[indices].detach().cpu()
            p_kpts_l = out_kpts_l[batch_idx, indices].detach().cpu()
            p_kpts_r = out_kpts_r[batch_idx, indices].detach().cpu()

            # sort the top for visualization
            topk['box_l'].append(boxes_l[batch_idx, indices])
            topk['box_r'].append(boxes_r[batch_idx, indices])
            topk['kpts_l'].append(out_kpts_l[batch_idx, indices])
            topk['kpts_r'].append(out_kpts_r[batch_idx, indices])
            topk['loc_l'].append(sampling_loc_l[batch_idx, indices])
            topk['loc_r'].append(sampling_loc_r[batch_idx, indices])
            topk['weight_l'].append(atten_weight_l[batch_idx, indices])
            topk['weight_r'].append(atten_weight_r[batch_idx, indices])

            preds_l.append(torch.cat([p_box_l, p_score, p_label], dim=-1))
            pred_r.append(torch.cat([p_box_r, p_score, p_label], dim=-1))

            pred_kpts_l.append(p_kpts_l)
            pred_kpts_r.append(p_kpts_r)

        return preds_l, pred_r, pred_kpts_l, pred_kpts_r, topk
