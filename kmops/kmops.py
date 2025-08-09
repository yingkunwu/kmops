import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from utils.util import (
    NestedTensor, kpts_disp_to_left_right, reproject_2d_to_3d_batch,
    gen_reproj_matrix_batch
)
from utils.box import Box


class KMOPS(nn.Module):
    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.add_module("backbone", backbone)
        self.add_module("encoder", encoder)
        self.add_module("decoder", decoder)

    def forward(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
        - samples.tensor: batched stereo images,
                          each of shape [batch_size x 3 x H x W]
        - samples.mask: a binary mask of shape [batch_size x H x W],
                        containing 1 on padded pixels

        It returns a dict with the following elements:
          "pred_logits" [batch_size x num_queries x num_classes + 1]:
            the classification logits for all queries.
         "pred_kpts" [batch_size x num_queries x 8 x 2]:
            The normalized average keypoint coordinates for all queries,
            These values are normalized in [0, 1], relative to the size of
            each individual image. The actual left and right keypoints located
            at left and right side of the image can be obtained by
            `kpts_disp_to_left_right(pred_kpts, pred_disp)`.
          "pred_disp" [batch_size x num_queries x 1]:
            The normalized disparity for all queries, which is also normalized
            in [0, 1], relative to the size of each individual image.
          "pred_conf" [batch_size x num_queries x 1]:
            The visibility score for each keypoint.
        """
        src_l, src_r, mask = samples.decompose()
        features = self.backbone(torch.cat([src_l, src_r], dim=0))

        src_list, mask_list = [], []
        for layer, src in features.items():
            if layer == '0':
                continue
            m = mask.clone()
            m = F.interpolate(m[None].float(), size=src.shape[-2:])
            m = m[0].to(torch.bool)

            src_list.append(src)
            mask_list.append(m)

        src_list = self.encoder(src_list, mask_list)
        out = self.decoder(src_list, mask_list)

        return out

    @torch.no_grad()
    def postprocess(self, outputs, image_size, targets=None):
        """Post-process the model outputs to get predictions.

        Parameters:
            outputs: raw outputs of the model
                "pred_logits" [batch_size, num_queries, num_classes]
                "pred_kpts" [batch_size, num_queries, num_keypoints, 2]
                "pred_disp" [batch_size, num_queries, 1]
                "pred_conf" [batch_size, num_queries, 1]
            image_size: the original size of the images, used to convert
                the normalized coordinates to absolute pixel coordinates.
            targets: (optional) ground truth targets, used for training.
        """
        def to_cpu(tensor):
            return tensor.detach().cpu()

        out_logits = to_cpu(outputs['pred_logits'])
        out_kpts = to_cpu(outputs['pred_kpts'])
        out_disp = to_cpu(outputs['pred_disp'])
        out_conf = to_cpu(outputs['pred_conf'].sigmoid())
        sampling_loc_l = to_cpu(outputs['log_infos'][0])
        sampling_loc_r = to_cpu(outputs['log_infos'][2])
        atten_weight_l = to_cpu(outputs['log_infos'][1])
        atten_weight_r = to_cpu(outputs['log_infos'][3])

        kpts_l, kpts_r = kpts_disp_to_left_right(out_kpts, out_disp)

        assert len(image_size) == len(out_logits), \
            "image_size should have the same length as outputs"
        assert len(image_size[0]) == 2, \
            "image_size should be a tuple (height, width)"
        image_size = torch.as_tensor(image_size, device=out_logits.device)
        image_size = image_size[:, [1, 0]]  # convert to (width, height)

        results = {
            "score": [],
            "label": [],
            "kpts_l": [],
            "kpts_r": [],
            "conf": [],
            "loc_l": [],
            "loc_r": [],
            "weight_l": [],
            "weight_r": [],
        }

        # convert realtive [0, 1] to absolute coordinates
        kpts_l *= rearrange(image_size, 'b n -> b 1 1 n')
        kpts_r *= rearrange(image_size, 'b n -> b 1 1 n')
        sampling_loc_l *= rearrange(image_size, 'b n -> b 1 1 n')
        sampling_loc_r *= rearrange(image_size, 'b n -> b 1 1 n')

        for batch_idx in range(out_logits.size(0)):
            prob = out_logits[batch_idx].sigmoid()
            scores, labels = prob.max(-1, keepdim=True)

            _, indices = torch.topk(
                scores.flatten(), 100, largest=True, sorted=True)

            # sort the top for visualization
            results['score'].append(scores[indices, 0])
            results['label'].append(labels[indices, 0])
            results['kpts_l'].append(kpts_l[batch_idx, indices])
            results['kpts_r'].append(kpts_r[batch_idx, indices])
            results['conf'].append(out_conf[batch_idx, indices])
            results['loc_l'].append(sampling_loc_l[batch_idx, indices])
            results['loc_r'].append(sampling_loc_r[batch_idx, indices])
            results['weight_l'].append(atten_weight_l[batch_idx, indices])
            results['weight_r'].append(atten_weight_r[batch_idx, indices])

        if targets is not None:
            results["gt_kpts_l"] = []
            results["gt_kpts_r"] = []
            results["gt_label"] = []

            for batch_idx in range(len(targets['labels'])):
                kpts = to_cpu(targets['kpts'][batch_idx])
                disp = to_cpu(targets['disp'][batch_idx])
                label = to_cpu(targets['labels'][batch_idx])

                kpts_l, kpts_r = kpts_disp_to_left_right(kpts, disp)
                kpts_l *= rearrange(image_size[batch_idx], 'n -> 1 1 n')
                kpts_r *= rearrange(image_size[batch_idx], 'n -> 1 1 n')

                results['gt_kpts_l'].append(kpts_l)
                results['gt_kpts_r'].append(kpts_r)
                results['gt_label'].append(label)

        return results

    @torch.no_grad()
    def reprojection(self, PL, PR, baseline, k2d_l, k2d_r):
        """
        Reproject 2D keypoints to 3D using the projection matrices and
        stereo baseline.

        Args:
            PL: Left camera projection matrix, shape [3, 4]
            PR: Right camera projection matrix, shape [3, 4]
            baseline: Baseline distance between the two cameras, shape [1]
            k2d_l: Left keypoints in 2D, shape [N, num_keypoints, 2]
            k2d_r: Right keypoints in 2D, shape [N, num_keypoints, 2]

        Returns:
            k3d: Reprojected 3D keypoints, shape [N, num_keypoints, 3]
        """
        Q = gen_reproj_matrix_batch(PL, PR, baseline)
        k3d = reproject_2d_to_3d_batch(k2d_l, k2d_r, Q)
        return k3d

    @torch.no_grad()
    def posefitting(self, k3d, conf=None):
        """
        Fit the pose of the 3D keypoints to get the sizes and poses.

        Args:
            k3d: 3D keypoints, shape [N, num_keypoints, 3]
            conf (optional): Confidence scores for the keypoints,
                             shape [N, num_keypoints, 1]
        Returns:
            A dictionary containing:
                "poses": list of N 4x4 homogeneous pose matrices,
                         shape [N, 4, 4]
                "scales": list of N box sizes [length, width, height],
                          shape [N, 3]
                "box3ds": list of N sets of 8 corner coordinates,
                          shape [N, 8, 3]
                "ax3ds": list of N axis endpoints in 3D, shape [N, 4, 3]
        """
        poses, scales = [], []
        box3ds, ax3ds = [], []
        for k, pt in enumerate(k3d):
            if conf is not None:
                box = Box.from_keypoints(pt, conf[k])
            else:
                box = Box.from_keypoints(pt)
            R, t = box.get_pose()
            # Build 4x4 homogeneous matrix
            T = torch.eye(4, dtype=R.dtype, device=R.device)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T.tolist())
            scales.append(box.get_size().tolist())

            box3d = box.get_keypoints(num_k=8).to(torch.float32)
            box3ds.append(box3d.tolist())

            # Define the pose axis in 3D space
            axis_length = 0.1
            axis = torch.tensor([
                [0, 0, 0],  # Origin
                [axis_length, 0, 0],  # X-axis
                [0, axis_length, 0],  # Y-axis
                [0, 0, axis_length]   # Z-axis
            ], dtype=torch.float64)
            ax3d = (R @ axis.T).T + t
            ax3d = ax3d.to(torch.float32)
            ax3ds.append(ax3d.tolist())

        return {
            "poses": poses,
            "scales": scales,
            "box3ds": box3ds,
            "ax3ds": ax3ds,
        }
