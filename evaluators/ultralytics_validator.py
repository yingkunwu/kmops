# Modified from https://github.com/ultralytics/ultralytics

import numpy as np
import torch

from .metrics import DetMetrics, PoseMetrics
from utils.util import box_iou, kpt_iou, box_xyxy_to_cxcywh


class DetectionValidator:
    """
    Attributes:
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        nc: Number of classes.
        iouv: (torch.Tensor):
            IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        nt_per_class (np.ndarray): Number of ground truth instances per class.
        nt_per_image (np.ndarray): Number of images that contain at least one
            instance of each class.

    """

    def __init__(self, names):
        self.names = names
        self.nc = len(names)
        self.seen = 0
        self.stats = None
        self.jdict = None
        self.nt_per_class = None
        self.nt_per_image = None
        self.metrics = DetMetrics()
        # IoU vector for mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

    def get_desc(self):
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)"
        )

    def match_predictions(self, pred_classes, true_classes, iou):
        """
        Matches predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU
                                values for predictions and ground of truth

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU
                            thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros(
            (pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[
                        iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[
                        np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[
                        np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True

        return torch.tensor(
            correct, dtype=torch.bool, device=pred_classes.device)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing
                detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing
                ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target
                class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10
                IoU levels.

        Note:
            The function does not return any value directly usable for metrics
            calculation. Instead, it provides an intermediate representation
            used for evaluating predictions against ground truth.
        """
        iou, _ = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def init_metrics(self):
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[],
                          target_img=[])
        self.jdict = []
        self.seen = 0

    def update_metrics(self, preds, targets):
        """
        Update metrics with predictions and targets.

        Args:
            preds (torch.Tensor):
                List of predictions tensors of shape (N, 6),
                where each prediction is (x1, y1, x2, y2, conf, class).
            targets (torch.Tensor):
                List of targets tensors of shape (M, 5),
                where each target is (x1, y1, x2, y2, class).

        """
        for p, t in zip(preds, targets):
            self.seen += 1
            npr = len(p)
            stat = dict(
                conf=torch.zeros(0, device=p.device),
                pred_cls=torch.zeros(0, device=p.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool,
                               device=p.device),
            )

            t_label = t[:, 4]
            nl = len(t_label)
            stat["target_cls"] = t_label
            stat["target_img"] = t_label.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                return

            # Predictions
            stat["conf"] = p[:, 4]
            stat["pred_cls"] = p[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(p, t[:, :4], t_label)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy()
                 for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(
            stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        print(self.get_desc())

        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
        print(pf % ("all", self.seen, self.nt_per_class.sum(),
                    *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            print(f"WARNING: no labels found in {self.args.task} set, "
                  "can not compute metrics without labels")
        # Print results per class
        if self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                print(pf % (self.names[c], self.nt_per_image[c],
                            self.nt_per_class[c],
                            *self.metrics.class_result(i)))

    def get_result(self):
        stats = self.get_stats()
        self.print_results()
        # return results as 5 decimal place floats
        res = {k: round(float(v), 5) for k, v in stats.items()}
        res.pop("fitness", None)
        return res


class PoseValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a
    pose model.
    """

    def __init__(self, names, num_keypoints):
        super().__init__(names)
        self.metrics = PoseMetrics()
        self.sigmas = torch.ones(num_keypoints, dtype=torch.float32) / 10

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self):
        """Initiate pose estimation metrics for YOLO model."""
        super().init_metrics()
        self.stats = dict(tp=[], tp_p=[], conf=[], pred_cls=[], target_cls=[],
                          target_img=[])

    def update_metrics(self, preds, targets, pred_kpts, tart_kpts):
        """
        Update metrics with predictions and targets.

        Args:
            preds (torch.Tensor):
                List of predictions tensors of shape (N, 6),
                where each prediction is (x1, y1, x2, y2, conf, class).
            targets (torch.Tensor):
                List of targets tensors of shape (M, 5),
                where each target is (x1, y1, x2, y2, class).
            pred_kpts (torch.Tensor):
                List of predicted keypoints tensors of shape (N, num_kpts, 2)
                where each keypoint is (x, y).
            tart_kpts (torch.Tensor):
                List of target keypoints tensors of shape (M, num_kpts, 2) or
                (M, num_kpts, 3) where each keypoint is (x, y) or (x, y, v).

        """
        for p, t, pk, tk in zip(preds, targets, pred_kpts, tart_kpts):
            self.seen += 1
            npr = len(p)
            stat = dict(
                conf=torch.zeros(0, device=p.device),
                pred_cls=torch.zeros(0, device=p.device),
                tp=torch.zeros(
                    npr, self.niou, dtype=torch.bool, device=p.device),
                tp_p=torch.zeros(
                    npr, self.niou, dtype=torch.bool, device=p.device),
            )

            t_label = t[:, 4]
            nl = len(t_label)
            stat["target_cls"] = t_label
            stat["target_img"] = t_label.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                return

            # Predictions
            stat["conf"] = p[:, 4]
            stat["pred_cls"] = p[:, 5]

            # Evaluate
            if nl:
                stat["tp"], stat["tp_p"] = \
                    self._process_batch(p, t[:, :4], t_label, pk, tk)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def _process_batch(self, dets, gt_bboxes, gt_cls, pred_kpts, gt_kpts):
        """
        Return correct prediction matrix by computing Intersection over Union
        (IoU) between detections and ground truth.

        Args:
            dets (torch.Tensor): Tensor with shape (N, 6) representing
                detection boxes and scores, where each detection is of the
                format (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor with shape (M, 4) representing
                ground truth bounding boxes, where each box is of the format
                (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor with shape (M,) representing ground
                truth class indices.
            pred_kpts (torch.Tensor): A tensor of shape (N, 17, 2) representing
                ground truth keypoints.
            gt_kpts (torch.Tensor): A tensor of shape (M, 17, 3) representing
                predicted keypoints, where the third dimension indicates the
                visibility of the keypoints.

        Returns:
            (torch.Tensor): A tensor with shape (N, 10) representing the
                correct prediction matrix for 10 IoU levels,
                where N is the number of detections.

        Note:
            The 0.53 multiplier comes from the way a 2D Gaussian function
            distributes its probability over space. In pose estimation,
            we often model keypoints (like joints in human pose detection)
            using a Gaussian to represent the uncertainty in their location.
            When integrating a Gaussian with a standard deviation of 0.25 over
            its entire area, the result approximates 0.53. This means that
            about 53% of the total probability mass is contained within a
            standard region around the keypoint. Since different objects have
            different sizes, we need a way to normalize the area considered
            around each keypoint so that models are compared fairly, no matter
            how big or small the objects are. The Object Keypoint Similarity
            metric, which measures how well a model's predicted keypoints match
            the ground truth, uses this 0.53 scaling factor to ensure that
            keypoints are evaluated consistently across objects of different
            scales. This way, a model isn't unfairly penalized just because
            it's working with larger or smaller objects—everything is compared
            on the same relative scale.
        """
        area = box_xyxy_to_cxcywh(gt_bboxes)[:, 2:].prod(1) * 0.53
        sigma = self.sigmas.to(pred_kpts.device)

        iou, _ = box_iou(gt_bboxes, dets[:, :4])
        tp = self.match_predictions(dets[:, 5], gt_cls, iou)
        iou = kpt_iou(gt_kpts, pred_kpts, sigma=sigma, area=area)
        tp_p = self.match_predictions(dets[:, 5], gt_cls, iou)

        return tp, tp_p
