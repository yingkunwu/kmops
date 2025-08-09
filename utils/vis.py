import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.optimize import linear_sum_assignment
from einops import rearrange
from sklearn.decomposition import PCA
from io import BytesIO

from utils.util import (
    to_numpy, to_tensor, project_3d_to_2d_batch, kpts_to_boxes
)

matplotlib.use('Agg')


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
    proj_mat_l, proj_mat_r, save_path=None, normalize_image=False
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

    if normalize_image:
        img_l = norm_image(img_l)
        img_r = norm_image(img_r)
        # convert tensor to numpy
        img_l = torch_img_to_numpy_img(img_l)
        img_r = torch_img_to_numpy_img(img_r)
        # convert RGB to BGR for OpenCV
        img_l = img_l[..., ::-1].copy()
        img_r = img_r[..., ::-1].copy()

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

    if save_path is not None:
        cv2.imwrite(save_path, combined)

    return combined


def visualize(
    img_l, img_r, pred_box3ds, pred_scores, pred_class_ids, pred_ax3ds,
    pred_kpts_l, pred_kpts_r, class_names, proj_mat_l, proj_mat_r,
    save_path=None, normalize_image=False
):
    """
    Draw prediction boxes, ground-truth (red) boxes and axes,
    then save a side-by-side image.
    """
    pred_box3ds = to_tensor(pred_box3ds)
    pred_scores = to_tensor(pred_scores)
    pred_ax3ds = to_tensor(pred_ax3ds)

    if normalize_image:
        img_l = norm_image(img_l)
        img_r = norm_image(img_r)
        # convert tensor to numpy
        img_l = torch_img_to_numpy_img(img_l)
        img_r = torch_img_to_numpy_img(img_r)
        # convert RGB to BGR for OpenCV
        img_l = img_l[..., ::-1].copy()
        img_r = img_r[..., ::-1].copy()

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

    if save_path is not None:
        cv2.imwrite(save_path, combined)

    return combined


def norm_image(image):
    min = float(image.min())
    max = float(image.max())
    image.add_(-min).div_(max - min + 1e-5)
    return image


def norm_atten_map(atten_map):
    atten_map = (
        (atten_map - np.min(atten_map))
        / (np.max(atten_map) - np.min(atten_map))
    )
    return atten_map


def torch_img_to_numpy_img(image):
    # convert tensor to numpy
    image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0)
    image = to_numpy(image)
    return image


def save_attention_map(
        batch_image, batch_attn, batch_out, file_name,
        normalize_image=True, normalize_atten_map=True, num_display=4):

    b, _, h, w = batch_image.shape
    num_objs = batch_out[0].size(0)

    if normalize_image:
        batch_image = norm_image(batch_image)
        # resize image
        batch_image = cv2.resize(
            batch_image.copy(),
            (batch_image.shape[1]//4, batch_image.shape[0]//4))

    fig = plt.figure(figsize=(30, 30))
    fig.subplots_adjust(
        bottom=0.02, right=0.97, top=0.98, left=0.03,
    )

    outer = gridspec.GridSpec(1, num_display, wspace=0.15)

    for b in range(num_display):
        image = torch_img_to_numpy_img(batch_image[b])
        attn_map = batch_attn[b]
        obj_box = to_numpy(batch_out[b])

        # reshape attention map to match the image size (approximately)
        feat_h = h // 32 + 1 if h % 32 != 0 else h // 32
        feat_w = w // 32 + 1 if w % 32 != 0 else w // 32
        attn_map = rearrange(attn_map, 'n (h w) -> n h w', h=feat_h, w=feat_w)

        inner = gridspec.GridSpecFromSubplotSpec(
            num_objs + 1, 1,
            subplot_spec=outer[b],
            wspace=0.001, hspace=0.05)

        ax = plt.Subplot(fig, inner[0])

        ax.set_xlabel(f"sample_{b}", fontsize=20)
        ax.xaxis.set_label_position('top')
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

        for j in range(num_objs):
            ax = plt.Subplot(fig, inner[j + 1])
            ax.imshow(image)

            attn_map_obj = F.interpolate(
                attn_map[None, None, j, :, :],
                scale_factor=8,
                mode="bilinear").squeeze()
            attn_map_obj = to_numpy(attn_map_obj)

            if normalize_atten_map:
                attn_map_obj = norm_atten_map(attn_map_obj)
            im = ax.imshow(attn_map_obj, cmap="nipy_spectral", alpha=0.7)

            box = obj_box[j] / 4
            rect = plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    cax = plt.axes([0.975, 0.025, 0.005, 0.95])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.0, 0.5, 1])
    cb.ax.tick_params(labelsize=20)
    plt.savefig(file_name)
    plt.close()


def save_attention_loc(batch_image_l, batch_image_r, batch_outputs,
                       normalize_image=True, num_display=4, num_objs=3):
    if normalize_image:
        batch_image_l = norm_image(batch_image_l)
        batch_image_r = norm_image(batch_image_r)

    fig = plt.figure(figsize=(30, 30))
    fig.subplots_adjust(
        bottom=0.02, right=0.97, top=0.98, left=0.03,
    )

    batch_kpts_l = batch_outputs["kpts_l"]
    batch_kpts_r = batch_outputs["kpts_r"]
    batch_loc_l = batch_outputs["loc_l"]
    batch_loc_r = batch_outputs["loc_r"]
    batch_weight_l = batch_outputs["weight_l"]
    batch_weight_r = batch_outputs["weight_r"]

    batch_image = [batch_image_l, batch_image_r]
    batch_kpts = [batch_kpts_l, batch_kpts_r]
    batch_loc = [batch_loc_l, batch_loc_r]
    batch_weight = [batch_weight_l, batch_weight_r]

    outer = gridspec.GridSpec(1, num_display * 2, wspace=0.15)

    for b in range(num_display * 2):
        image = torch_img_to_numpy_img(batch_image[b % 2][b // 2])
        image = cv2.resize(
            image.copy(), (image.shape[1]//4, image.shape[0]//4))
        sample_location = to_numpy(batch_loc[b % 2][b // 2])
        attn_weight = to_numpy(batch_weight[b % 2][b // 2])

        obj_kpts = batch_kpts[b % 2][b // 2]
        obj_box = kpts_to_boxes(obj_kpts)
        obj_kpts = to_numpy(obj_kpts)
        obj_box = to_numpy(obj_box)

        inner = gridspec.GridSpecFromSubplotSpec(
            num_objs + 1, 1,
            subplot_spec=outer[b],
            wspace=0.001, hspace=0.05)

        ax = plt.Subplot(fig, inner[0])

        ax.set_xlabel(f"sample_{b // 2}", fontsize=20)
        ax.xaxis.set_label_position('top')
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

        vmin, vmax = attn_weight.min(), attn_weight.max()

        for j in range(num_objs):
            ax = plt.Subplot(fig, inner[j + 1])
            ax.imshow(image)

            # Normalize weights for color mapping
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap("coolwarm")  # Red for high, blue for low
            colors = [cmap(norm(w)) for w in attn_weight[j]]

            y = [int(loc[1] / 4) for loc in sample_location[j]]
            x = [int(loc[0] / 4) for loc in sample_location[j]]
            ax.scatter(x, y, c=colors, s=20)

            box = obj_box[j] / 4
            rect = plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='r', facecolor='none')

            kpts = obj_kpts[j] / 4
            ax.scatter(kpts[:, 0], kpts[:, 1], c='lime', s=20)

            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    cax = plt.axes([0.975, 0.025, 0.005, 0.95])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([vmin, vmax])
    cb.ax.tick_params(labelsize=20)

    plt.close()
    return fig


def visualize_pca_features(feats, dim, target_size, figsize=(6, 6)):
    """
    PCA-based feature-map visualization.

    Args:
        feats (torch.Tensor):
            [C, H, W] or [B, C, H, W]. If batched, we visualize only the first
            sample.
        dim (int):
            number of principal components to keep (must be >=3 to show RGB).
        fit_pca (sklearn.decomposition.PCA, optional):
            if provided, use this pre-fitted PCA; otherwise fit a new PCA on
            this map.
        figsize (tuple): size of the matplotlib figure.
    """
    # take first element if batched
    if feats.dim() == 4:
        feats = feats[0]

    C, H, W = feats.shape
    feats = F.interpolate(
        feats.unsqueeze(0), size=target_size, mode="bilinear",
        align_corners=False).squeeze(0)  # [C, H, W]
    C, H, W = feats.shape

    # flatten to shape [H*W, C]
    x = feats.view(C, -1).permute(1, 0).cpu().numpy()  # (N_pixels, C)

    # fit PCA if needed
    fit_pca = PCA(n_components=dim)
    fit_pca.fit(x)

    # project down
    x_red = fit_pca.transform(x)  # (N_pixels, dim)

    # reshape back to [dim, H, W]
    x_red = torch.from_numpy(x_red).view(H, W, dim).permute(2, 0, 1)

    # normalize each component to [0,1]
    mins = x_red.view(dim, -1).min(dim=1)[0].view(dim, 1, 1)
    maxs = x_red.view(dim, -1).max(dim=1)[0].view(dim, 1, 1)
    img = (x_red - mins) / (maxs - mins + 1e-6)

    # if dim > 3, only take the first 3 for RGB
    rgb = img[:3].permute(1, 2, 0).cpu()

    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(rgb)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    # Convert the BytesIO object to a numpy array
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image
