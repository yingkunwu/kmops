import numpy as np
import torch
from torchvision.ops.boxes import box_area


def to_tensor(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        x = x.type(dtype)
    elif isinstance(x, list):
        x = np.array(x)
        x = torch.tensor(x, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        if x.dtype != dtype:
            x = x.type(dtype)
    else:
        raise TypeError(f"Unsupported type {type(x)}")

    return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, list):
        raise TypeError(f"Unsupported type {type(x)}")

    return x


class NestedTensor(object):
    def __init__(self, image_l, image_r, mask):
        self.image_l = image_l
        self.image_r = image_r
        self.mask = mask

    def to(self, device):
        cast_image_l = self.image_l.to(device)
        cast_image_r = self.image_r.to(device)
        cast_mask = self.mask.to(device)
        return NestedTensor(cast_image_l, cast_image_r, cast_mask)

    def decompose(self):
        return self.image_l, self.image_r, self.mask

    def __repr__(self):
        return f"NestedTensor(image_l=tensor({self.image_l.shape}), " \
               f"image_r=tensor({self.image_r.shape}), " \
               f"mask=tensor({self.mask.shape}))"


def kpts_disp_to_left_right(midpoints, disparities):
    # turn 1‐channel disparity into 2D vector and split left/right
    disp_vec = torch.cat([disparities, torch.zeros_like(disparities)], dim=-1)
    half = disp_vec.mul(0.5)
    return midpoints[..., :2] + half, midpoints[..., :2] - half


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def kpts_to_boxes(kpts):
    x1 = kpts[..., 0].min(dim=-1).values
    y1 = kpts[..., 1].min(dim=-1).values
    x2 = kpts[..., 0].max(dim=-1).values
    y2 = kpts[..., 1].max(dim=-1).values
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union of boxes1 and boxes2
    The boxes should be in [x0, y0, x1, y1] format

    Args:
        boxes1: (tensor) bounding boxes, sized [N, 4]
        boxes2: (tensor) bounding boxes, sized [M, 4]
    Returns:
        (tensor) iou, sized [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area  # giou


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor):
            A tensor of shape (N, 17, 3) representingground truth keypoints.
        kpt2 (torch.Tensor):
            A tensor of shape (M, 17, 2) representing predicted keypoints.
        area (torch.Tensor):
            A tensor of shape (N,) representing areas from ground truth.
        sigma (list):
            A list containing 17 values representing keypoint scales.
        eps (float, optional):
            A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor):
            A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) \
        + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    kpt_mask = kpt1[..., 2] > 0  # (N, 17)
    # from cocoeval
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) \
        / (kpt_mask.sum(-1)[:, None] + eps)


def project_3d_to_2d_batch(
    pts_3d: torch.Tensor,
    P: torch.Tensor
) -> torch.Tensor:
    """
    Project 3D points to 2D using the camera projection matrix.

    Args:
        pts_3d (torch.Tensor): 3D points, shape [N, 3] or [B, N, 3].
        P (torch.Tensor): Camera projection matrix, shape [3, 4].

    Returns:
        torch.Tensor: 2D points, shape [N, 2] or [B, N, 2].
    """
    if pts_3d.dim() not in (2, 3):
        raise ValueError("pts_3d must be of shape (N,3) or (B,N,3)")

    *batch_dims, N, _ = pts_3d.shape
    ones = torch.ones(*batch_dims, N, 1,
                      device=pts_3d.device,
                      dtype=pts_3d.dtype)
    homo = torch.cat([pts_3d, ones], dim=-1)
    pts_proj = homo @ P.T
    x = pts_proj[..., 0]
    y = pts_proj[..., 1]
    z = pts_proj[..., 2]
    pts_2d = torch.stack([x / z, y / z], dim=-1)
    return pts_2d


def reproject_2d_to_3d_batch(
    kpt_l: torch.Tensor,
    kpt_r: torch.Tensor,
    Q: torch.Tensor
) -> torch.Tensor:
    """
    Reproject 2D keypoints to 3D using disparity.

    Args:
        kpt_l (torch.Tensor): Left 2D keypoints, shape [B, N, 2].
        kpt_r (torch.Tensor): Right 2D keypoints, shape [B, N, 2].
        Q (torch.Tensor): Disparity-to-depth mapping matrices,
                          shape [4, 4] or [B, 4, 4].

    Returns:
        torch.Tensor: 3D keypoints, shape [B, N, 3].
    """
    assert kpt_l.shape == kpt_r.shape and kpt_l.dim() == 3 \
        and kpt_l.size(-1) == 2, \
        f"Invalid keypoint shapes {kpt_l.shape} and {kpt_r.shape}, " \
        "expected [B, N, 2]"
    # ensure Q is [B, 4, 4]
    assert Q.shape[-2:] == (4, 4), \
        f"Invalid Q shape {Q.shape}, expected 4 by 4"

    if Q.dim() == 2:
        B = kpt_l.shape[0]
        Q = Q.repeat(B, 1, 1)

    # compute disparity (x difference)
    disp = kpt_l[..., :1] - kpt_r[..., :1]
    # build homogeneous coords [x, y, disp, 1]
    hom_kpt = torch.cat([kpt_l, disp, torch.ones_like(disp)], dim=-1)
    # project through Q
    pts4d = torch.bmm(Q, hom_kpt.transpose(1, 2)).transpose(1, 2)
    # normalize
    return pts4d[..., :3] / pts4d[..., 3:]


def gen_reproj_matrix_batch(
    P_left: torch.Tensor,
    P_right: torch.Tensor,
    baseline: torch.Tensor
) -> torch.Tensor:
    """
    Create batch disparity-to-depth mapping matrices (Q matrices).

    Args:
        P_left (torch.Tensor): Left camera projection matrix,
            shape [3, 4] or [B, 3, 4].
        P_right (torch.Tensor): Right camera projection matrix,
            shape [3, 4] or [B, 3, 4].
        baseline (torch.Tensor): Distance between cameras,
            shape [1] or [B].

    Returns:
        torch.Tensor: Disparity-to-depth mapping matrices,
            shape [4, 4] or [B, 4, 4].
    """
    assert (
        (P_left.dim() == 2 and P_right.dim() == 2 and baseline.dim() == 1)
        or (P_left.dim() == 3 and P_right.dim() == 3 and baseline.dim() == 2)
    ), f"Invalid shapes: P_left {P_left.shape}, P_right {P_right.shape}, "\
       f"baseline {baseline.shape}"

    # ensure inputs are batched
    batched = True
    if P_left.dim() == 2:
        P_left = P_left.unsqueeze(0)
        P_right = P_right.unsqueeze(0)
        baseline = baseline.unsqueeze(0)
        batched = False

    B = P_left.shape[0]

    # initialize Q_batch = eye(4) for each batch
    Q = torch.eye(4, device=P_left.device, dtype=P_left.dtype)\
        .unsqueeze(0).repeat(B, 1, 1)

    # fill according to cv2.stereoRectify Q definition
    Q[:, 0, 3] = -P_left[:, 0, 2]
    Q[:, 1, 3] = -P_left[:, 1, 2]
    Q[:, 2, 3] = P_left[:, 0, 0]
    Q[:, 2, 2] = 0.0
    Q[:, 3, 2] = 1.0 / baseline
    Q[:, 3, 3] = -(P_left[:, 0, 2] - P_right[:, 0, 2]) / baseline

    return Q if batched else Q.squeeze(0)


def get_min_vol_ellipse(
    P_batch: torch.Tensor,
    tol: float = 0.01,
    max_iters: int = 1000
) -> tuple:
    """
    Find the minimum-area ellipse which encloses all 2D points in P.

    Args:
        P_batch: Tensor of shape (B, N, 2) or (N, 2)
        tol:     convergence tolerance

    Returns:
        centers:   (B, 2) or (2,)
        radii:     (B, 2) or (2,)
        rotations: (B, 2, 2) or (2, 2)
    """
    P = P_batch
    if P.dim() == 2:
        P = P.unsqueeze(0)
    B, N, d = P.shape
    assert d == 2, "Input must be (B,N,2) or (N,2)"

    ones = P.new_ones((B, 1, N))
    Q = torch.cat((P.permute(0, 2, 1), ones), dim=1)  # (B,3,N)

    # Initialize uniform weights u_i = 1/N
    u = P.new_full((B, N), 1.0 / N)

    for _ in range(max_iters):
        V = torch.bmm(Q, (Q * u.unsqueeze(1)).transpose(1, 2))
        V_inv = torch.linalg.inv(V)

        end = torch.bmm(V_inv, Q)
        M = torch.sum(Q * end, dim=1)

        max_M, j = M.max(dim=1)

        # Step size per batch
        step = (max_M - d - 1.0) / ((d + 1) * (max_M - 1.0))
        step = step.clamp(min=0.0)

        # Update u
        new_u = (1.0 - step).unsqueeze(1) * u
        new_u = new_u.scatter_add(1, j.unsqueeze(1), step.unsqueeze(1))

        # Check convergence
        if torch.max(torch.norm(new_u - u, dim=1)) <= tol:
            u = new_u
            break
        u = new_u

    # Compute center: sum_i u_i * P_i
    centers = torch.sum(u.unsqueeze(2) * P, dim=1)  # (B,2)

    # Compute shape matrix A = inv( ∑ u_i x_i x_i^T − c c^T ) / d
    weighted = u.unsqueeze(2) * P
    S = torch.bmm(P.permute(0, 2, 1), weighted)
    ccT = centers.unsqueeze(2) @ centers.unsqueeze(1)
    A = torch.linalg.inv(S - ccT) / float(d)

    # Eigen‐decomposition A = Q diag(w) Q^T
    w, Qe = torch.linalg.eigh(A)
    radii = 1.0 / torch.sqrt(w)
    rotations = Qe

    if P_batch.dim() == 2:
        return centers[0], radii[0], rotations[0]
    return centers, radii, rotations
