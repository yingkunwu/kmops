import numpy as np
import torch

from utils.util import (
    project_3d_to_2d_batch, reproject_2d_to_3d_batch, gen_reproj_matrix_batch
)
from utils.box import Box


class Inferencer:
    def __init__(self, model, postprocessor, device):
        self.device = device
        self.model = model.eval().to(device)
        self.postprocessor = postprocessor

    def estimate(self, samples, targets, use_gt=False):
        with torch.no_grad():
            out = self.model(samples.to(self.device))
        preds_l, preds_r, kpts_l, kpts_r, _ = \
            self.postprocessor(out, targets["ori_shape"])
        pred_conf = out['pred_conf'].sigmoid().detach().cpu()

        results = []  # will hold per-sample dicts
        for i in range(len(kpts_l)):
            # if use_gt, we only keep the top k predictions,
            # where k is the number of objects in the ground truth
            if use_gt:
                num_obj = targets['kpts_3d'][i].shape[0]
            else:
                num_obj = 100

            scores = preds_l[i][:num_obj, -2]
            labels = preds_l[i][:num_obj, -1]
            k2d_l = kpts_l[i][:num_obj]
            k2d_r = kpts_r[i][:num_obj]
            conf = pred_conf[i][:num_obj]

            # decode & project bboxes and axes
            PL = targets['proj_matrix_l'][i].to(torch.float32)
            PR = targets['proj_matrix_r'][i].to(torch.float32)
            baseline = targets['baseline'][i].to(torch.float32)

            # 3D reproject
            Q = gen_reproj_matrix_batch(PL, PR, baseline)
            k3d = reproject_2d_to_3d_batch(k2d_l, k2d_r, Q)

            pred_l, pred_r, pred_pose, axes_l, axes_r = [], [], [], [], []
            pred_size, gt_size = [], []
            for k, pt in enumerate(k3d):
                box = Box.from_keypoints(pt, conf[k])
                R, t = box.get_pose()

                pred_pose.append(torch.cat((R, t.reshape(3, 1)), dim=1))

                kpts = box.get_keypoints(num_k=8).to(torch.float32)
                pred_l.append(project_3d_to_2d_batch(kpts, PL))
                pred_r.append(project_3d_to_2d_batch(kpts, PR))

                # Define the pose axis in 3D space
                axis_length = 0.1  # Adjust the length of the axis as needed
                axis = torch.tensor([
                    [0, 0, 0],  # Origin
                    [axis_length, 0, 0],  # X-axis
                    [0, axis_length, 0],  # Y-axis
                    [0, 0, axis_length]   # Z-axis
                ], dtype=torch.float64)
                ax3d = (R @ axis.T).T + t
                ax3d = ax3d.to(torch.float32)
                axes_l.append(project_3d_to_2d_batch(ax3d, PL))
                axes_r.append(project_3d_to_2d_batch(ax3d, PR))
                pred_size.append(box.get_size().tolist())

            # ground truth
            gt_k3d = targets['kpts_3d'][i]
            visibility = targets["kpts_pose"][i][:, :, 2]
            gt_l, gt_r, gt_pose = [], [], []
            for pt in gt_k3d:
                box = Box.from_keypoints(pt)
                R, t = box.get_pose()
                gt_pose.append(torch.cat((R, t.reshape(3, 1)), dim=1))

                kpts = box.get_keypoints(num_k=8).to(torch.float32)
                gt_l.append(project_3d_to_2d_batch(kpts, PL))
                gt_r.append(project_3d_to_2d_batch(kpts, PR))
                gt_size.append(box.get_size().tolist())

            # Convert pred_pose from 3x4 to 4x4
            pred_pose = [
                torch.cat(
                    (pose, torch.tensor([[0, 0, 0, 1]], device=pose.device)),
                    dim=0
                )
                for pose in pred_pose
            ]
            gt_pose = [
                torch.cat(
                    (pose, torch.tensor([[0, 0, 0, 1]], device=pose.device)),
                    dim=0
                )
                for pose in gt_pose
            ]

            results.append({
                'pred_boxes_l': pred_l,
                'pred_boxes_r': pred_r,
                'pred_pose': pred_pose,
                'gt_boxes_l': gt_l,
                'gt_boxes_r': gt_r,
                'gt_pose': gt_pose,
                'k3d': k3d,
                'gt_k3d': gt_k3d,
                'k2d_l': k2d_l,
                'k2d_r': k2d_r,
                'conf': conf,
                'axes_l': axes_l,
                'axes_r': axes_r,
                'scores': scores,
                'visibility': visibility,
                "gt_class_ids": targets['labels'][i],
                "gt_scales": np.array(gt_size),
                "gt_RTs": gt_pose,
                "pred_class_ids": labels.long(),
                "pred_scales": np.array(pred_size),
                "pred_RTs": pred_pose,
                "pred_scores": scores
            })
        return results
