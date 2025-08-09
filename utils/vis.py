import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment

from utils.util import to_tensor, project_3d_to_2d_batch


colors_rgb = [
    (0, 255, 255),
    (162, 255, 0),
    (255, 230, 0),
    (170, 0, 255),
    (0, 255, 170),
    (85, 0, 255),
    (0, 255, 85),
    (255, 0, 0),
    (255, 0, 170),
    (49, 255, 0),
    (0, 170, 255),
    (255, 112, 0),
    (0, 85, 255),
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def plot_joints(img, joints, c=(0, 255, 0)):
    """
    joints: Nx2 or Nx3 array where each row is (x, y[, confidence])
    visibility: optional N-element boolean mask
    c: default circle/text color
    """
    joints = np.asarray(joints)
    num_joints, dims = joints.shape

    for k in range(num_joints):
        if dims > 2:
            x, y, conf = joints[k]
        else:
            x, y = joints[k]
            conf = None

        # choose color based on visibility
        color = c
        if conf is not None and conf < 0.5:
            color = (0, 0, 255)

        # draw the joint
        cv2.circle(img, (int(x), int(y)), 2, color, -1)

        # label the joint index
        cv2.putText(
            img, str(k), (int(x) + 2, int(y) - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

        # if confidence is provided, draw it
        if conf is not None:
            conf_text = f"{conf:.2f}"
            cv2.putText(
                img, conf_text, (int(x) + 2, int(y) + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    return img


def plot_3d_bbox(image, qs, thickness=1, color=[255, 255, 102]):
    """
    Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:

            z                    2 -------- 1
            |                   /|         /|
            |                  3 -------- 0 |
            |________ y        | |        | |
           /                   | 6 -------- 5
          /                    |/         |/
         x                     7 -------- 4
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)

    return image


def plot_axes(image, axis, thickness=1):
    """
    Draw 3D axes in the image.
    axes: (origin, x, y, z) where each is a 3D point.
    """
    origin, x, y, z = axis
    cv2.line(image, tuple(origin), tuple(x), (0, 0, 255), thickness)  # X-axis
    cv2.line(image, tuple(origin), tuple(y), (0, 255, 0), thickness)  # Y-axis
    cv2.line(image, tuple(origin), tuple(z), (255, 0, 0), thickness)  # Z-axis

    return image


def normalize_batch(img_batch, mean, std):
    """
    Denormalize a batch of images in CHW format.
    """
    mean = np.array(mean)[:, None, None]
    std = np.array(std)[:, None, None]
    imgs = img_batch.cpu().numpy() * std + mean
    imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)
    # from CHW to HWC
    return [cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            for img in imgs]


def match_predictions(src_kpts, tgt_kpts, src_scores):
    cost_kpts = (src_kpts[:, None, :] - tgt_kpts[None, :, :]) ** 2
    cost_kpts = torch.sqrt(torch.sum(cost_kpts + 1e-15, dim=-1))
    cost_kpts = torch.sum(cost_kpts, dim=-1)

    C = cost_kpts - src_scores[:, None] * 0.1
    return linear_sum_assignment(C.cpu().numpy())


def visualize_with_gt(
    img_l, img_r,
    pred_box3ds, pred_scores, pred_class_ids, pred_ax3ds,
    gt_box3ds, gt_class_ids, gt_ax3ds, class_names,
    proj_mat_l, proj_mat_r, save_path
):
    """
    Draw prediction boxes, ground-truth (red) boxes and axes,
    then save a side-by-side image.
    """
    pred_box3ds = to_tensor(pred_box3ds)
    pred_scores = to_tensor(pred_scores)
    pred_ax3ds = to_tensor(pred_ax3ds)
    gt_box3ds = to_tensor(gt_box3ds)
    gt_ax3ds = to_tensor(gt_ax3ds)

    img_l_gt = img_l.copy()
    img_r_gt = img_r.copy()

    i_src, i_tgt = match_predictions(pred_box3ds, gt_box3ds, pred_scores)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # helper to draw boxes, axes and labels on stereo images
    def _draw_on(img_l_i, img_r_i, box3ds, ax3ds, cls_ids, idxs):
        obj_labels = []
        for idx in idxs:
            box3d = box3ds[idx]
            ax3d = ax3ds[idx]
            label = cls_ids[idx].item()
            color = colors_rgb[label]

            # project and draw 3D bbox on left/right
            b2d_l = project_3d_to_2d_batch(box3d, proj_mat_l).numpy()
            b2d_r = project_3d_to_2d_batch(box3d, proj_mat_r).numpy()
            plot_3d_bbox(img_l_i, b2d_l, color=color)
            plot_3d_bbox(img_r_i, b2d_r, color=color)

            # project and draw axes
            a2d_l = project_3d_to_2d_batch(ax3d, proj_mat_l).to(int).numpy()
            a2d_r = project_3d_to_2d_batch(ax3d, proj_mat_r).to(int).numpy()
            plot_axes(img_l_i, a2d_l)
            plot_axes(img_r_i, a2d_r)

            text = class_names[label]
            xs, ys = b2d_l[:, 0], b2d_l[:, 1]
            x0, y0 = int(xs.min()), int(ys.min()) - 5
            obj_labels.append((text, color, x0, y0))

        for text, color, x0, y0 in obj_labels:
            (tw, th), baseline = cv2.getTextSize(text, font, 0.5, 1)
            tl = (x0, y0 - th - baseline)
            br = (x0 + tw + 4, y0 + 2)
            cv2.rectangle(img_l_i, tl, br, color, -1)
            cv2.putText(img_l_i, text, (x0 + 2, y0 - baseline + 1),
                        font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # draw predictions and ground-truths
    _draw_on(img_l, img_r, pred_box3ds, pred_ax3ds, pred_class_ids, i_src)
    _draw_on(img_l_gt, img_r_gt, gt_box3ds, gt_ax3ds, gt_class_ids, i_tgt)

    combined = np.hstack((img_l, img_r))
    combined_gt = np.hstack((img_l_gt, img_r_gt))
    cv2.putText(combined, "Predictions", (10, 20), font,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(combined_gt, "Ground Truth", (10, 20), font,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    combined = np.vstack((combined, combined_gt))
    cv2.imwrite(save_path, combined)


def visualize(
    img_l, img_r, pred_box3ds, pred_scores, pred_class_ids, pred_ax3ds,
    pred_kpts_l, pred_kpts_r, class_names, proj_mat_l, proj_mat_r, save_path
):
    """
    Draw prediction boxes, ground-truth (red) boxes and axes,
    then save a side-by-side image.
    """
    pred_box3ds = to_tensor(pred_box3ds)
    pred_scores = to_tensor(pred_scores)
    pred_ax3ds = to_tensor(pred_ax3ds)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # helper to draw boxes, axes and labels on stereo images
    def _draw_on(img_l_i, img_r_i, box3ds, ax3ds, cls_ids, idxs):
        obj_labels = []
        for idx in idxs:
            box3d = box3ds[idx]
            ax3d = ax3ds[idx]
            label = cls_ids[idx].item()
            score = pred_scores[idx].item()
            color = colors_rgb[label]

            # project and draw 3D bbox on left/right
            b2d_l = project_3d_to_2d_batch(box3d, proj_mat_l).numpy()
            b2d_r = project_3d_to_2d_batch(box3d, proj_mat_r).numpy()
            plot_3d_bbox(img_l_i, b2d_l, color=color)
            plot_3d_bbox(img_r_i, b2d_r, color=color)

            # project and draw axes
            a2d_l = project_3d_to_2d_batch(ax3d, proj_mat_l).to(int).numpy()
            a2d_r = project_3d_to_2d_batch(ax3d, proj_mat_r).to(int).numpy()
            plot_axes(img_l_i, a2d_l)
            plot_axes(img_r_i, a2d_r)

            # plot predicted keypoints on left and right images
            plot_joints(img_l_i, pred_kpts_l[idx])
            plot_joints(img_r_i,  pred_kpts_r[idx])

            # prepare label text with score
            text = f"{class_names[label]} {score:.2f}"
            xs, ys = b2d_l[:, 0], b2d_l[:, 1]
            x0, y0 = int(xs.min()), int(ys.min()) - 5
            obj_labels.append((text, color, x0, y0))

        for text, color, x0, y0 in obj_labels:
            (tw, th), baseline = cv2.getTextSize(text, font, 0.5, 1)
            tl = (x0, y0 - th - baseline)
            br = (x0 + tw + 4, y0 + 2)
            cv2.rectangle(img_l_i, tl, br, color, -1)
            cv2.putText(img_l_i, text, (x0 + 2, y0 - baseline + 1),
                        font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    i_src = torch.where(pred_scores > 0.5)[0]

    # draw predictions and ground-truths
    _draw_on(img_l, img_r, pred_box3ds, pred_ax3ds, pred_class_ids, i_src)
    combined = np.hstack((img_l, img_r))
    cv2.imwrite(save_path, combined)
