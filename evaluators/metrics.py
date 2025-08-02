import torch
import numpy as np


def mpjpe(output, target, visibility):
    error = torch.sqrt(
        torch.sum((output - target) ** 2 + 1e-15, dim=-1, keepdim=True))

    error *= visibility

    return torch.sum(error) / torch.sum(visibility)


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    # number of filter elements (must be odd)
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the
                      beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is
            correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1
            metric for each class. Shape: (nc,).
        fp (np.ndarray): False positive counts at threshold given by max F1
            metric for each class. Shape: (nc,).
        p (np.ndarray): Precision values at threshold given by max F1 metric
            for each class. Shape: (nc,).
        r (np.ndarray): Recall values at threshold given by max F1 metric for
            each class. Shape: (nc,).
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric
            for each class. Shape: (nc,).
        ap (np.ndarray): Average precision for each class at different IoU
            thresholds. Shape: (nc, 10).
        unique_classes (np.ndarray): An array of unique classes that have data.
            Shape: (nc,).
        p_curve (np.ndarray): Precision curves for each class.
            Shape: (nc, 1000).
        r_curve (np.ndarray): Recall curves for each class.
            Shape: (nc, 1000).
        f1_curve (np.ndarray): F1-score curves for each class.
            Shape: (nc, 1000).
        x (np.ndarray): X-axis values for the curves.
            Shape: (1000,).
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
            Shape: (nc, 1000).
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        # negative x, xp because xp decreases
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        # p at pr_score
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                # precision at mAP@0.5
                prec_values.append(np.interp(x, mrec, mpre))

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    # max-F1 precision, recall, F1 values
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), \
        p_curve, r_curve, f1_curve, x, prec_values


class Metric:
    """
    Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds.
            Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score.
            Shape: (nc,).

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes.
            Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes.
            Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes.
            Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes.
            Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes.
            Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result -> p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class.
            Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics.
            Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all
        classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per
                class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for
        all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per
                class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of
        0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5~0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """MAP of each class."""
        maps = self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation
                metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU
                    thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score.
                    Shape: (nc,).
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results


class DetMetrics:
    """
    Utility class for computing detection metrics such as precision, recall,
    and mean average precision (mAP) of an object detection model.

    Attributes:
        box (Metric): An instance of the Metric class for storing the results
            of the detection metrics.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results
            with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection
            metrics.
        mean_results: Returns a list of mean values for the computed detection
            metrics.
        class_result(i): Returns a list of values for the computed detection
            metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for
            different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection
            metrics.
        ap_class_index: Returns a list of class indices sorted by their
            average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to
            their computed values.
    """

    def __init__(self) -> None:
        """
        Initialize a DetMetrics instance with a save directory, plot flag,
        callback function, and class names.
        """
        self.box = Metric()

    def process(self, tp, conf, pred_cls, target_cls):
        """
        Process predicted results for object detection and update metrics.
        """
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
        )[2:]
        self.box.update(results)

    @property
    def keys(self):
        """
        Returns a list of keys for accessing specific metrics.
        """
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)"
        ]

    def mean_results(self):
        """
        Calculate mean of detected objects & return precision, recall,
        mAP50, and mAP50-95.
        """
        return self.box.mean_results()

    def class_result(self, i):
        """
        Return the result of evaluating the performance of an object
        detection model on a specific class.
        """
        return self.box.class_result(i)

    @property
    def maps(self):
        """
        Returns mean Average Precision (mAP) scores per class.
        """
        return self.box.maps

    @property
    def fitness(self):
        """
        Returns the fitness of box object.
        """
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """
        Returns the average precision index per class.
        """
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """
        Returns dictionary of computed performance metrics and statistics.
        """
        return dict(
            zip(self.keys + ["fitness"], self.mean_results() + [self.fitness])
        )


class PoseMetrics(DetMetrics):
    """
    Calculates and aggregates detection and pose metrics over a given set of
    classes.

    Attributes:
        box (Metric): An instance of the Metric class to calculate box
            detection metrics.
        pose (Metric): An instance of the Metric class to calculate mask
            segmentation metrics.

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls):
            Processes metrics over the given set of predictions.
        mean_results():
            Returns the mean of the detection and segmentation metrics over
            all the classes.
        class_result(i):
            Returns the detection and segmentation metrics of class `i`.
        maps:
            Returns the mean Average Precision (mAP) scores for IoU thresholds
            ranging from 0.50 to 0.95.
        fitness:
            Returns the fitness scores, which are a single weighted combination
            of metrics.
        ap_class_index:
            Returns the list of indices of classes used to compute Average
            Precision (AP).
        results_dict:
            Returns the dictionary containing all the detection and
            segmentation metrics and fitness score.
    """

    def __init__(self) -> None:
        super().__init__()
        self.box = Metric()
        self.pose = Metric()

    def process(self, tp, tp_p, conf, pred_cls, target_cls):
        """
        Processes the detection and pose metrics over the given set of
        predictions.
        """
        results_pose = ap_per_class(
            tp_p,
            conf,
            pred_cls,
            target_cls,
        )[2:]
        self.pose.update(results_pose)
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
        )[2:]
        self.box.update(results_box)

    @property
    def keys(self):
        """
        Returns list of evaluation metric keys.
        """
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self):
        """
        Return the mean results of box and pose.
        """
        return self.box.mean_results() + self.pose.mean_results()

    def class_result(self, i):
        """
        Return the class-wise detection results for a specific class i.
        """
        return self.box.class_result(i) + self.pose.class_result(i)

    @property
    def maps(self):
        """
        Returns the mean average precision (mAP) per class for both box and
        pose detections.
        """
        return self.box.maps + self.pose.maps

    @property
    def fitness(self):
        """
        Computes classification metrics and speed using the `targets` and
        `pred` inputs.
        """
        return self.box.fitness() + self.pose.fitness()
