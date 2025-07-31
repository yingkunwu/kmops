import numpy as np
import torch

from .util import to_tensor


def interpolate(bbox_3d, interp, interp_coef):
    """
    Interpolate 3D points on a 3D bounding box with 8 keypoints.
    Args:
        bbox_3d:  Tensor of shape (P, 3)
        interp:  A tuple (parent_indices, child_indices)
        interp_coef:  List of interpolation coefficients
    Returns:
        Tensor of shape (P + len(interp_coef) * len(parent_indices), 3)
    """
    # unpack list of [parent, child] pairs
    pidx = [pair[0] for pair in interp]
    cidx = [pair[1] for pair in interp]
    parents = bbox_3d[pidx]
    children = bbox_3d[cidx]
    directions = children - parents

    new_joints = []
    for p, d in zip(parents, directions):
        for coef in interp_coef:
            new_joints.append(p + coef * d)
    new_joints = torch.stack(new_joints, dim=0)

    return torch.cat([bbox_3d, new_joints], dim=0)


def get_canonical_box(size):
    length, width, height = size[0], size[1], size[2]

    # 3d bounding box corners
    x_corners = [length / 2, -length / 2, -length / 2, length / 2,
                 length / 2, -length / 2, -length / 2, length / 2]
    y_corners = [width / 2, width / 2, -width / 2, -width / 2,
                 width / 2, width / 2, -width / 2, -width / 2]
    z_corners = [height / 2, height / 2, height / 2, height / 2,
                 -height / 2, -height / 2, -height / 2, -height / 2]

    return torch.tensor([x_corners, y_corners, z_corners]).T


def decode_kpts_to_box(kps, conf=None):
    """
    kps: (N, 3) tensor of keypoints in 3D
    conf: (N,) or (N,1) tensor of non-negative weights.
    returns: [length, width, height]
    """
    n = kps.shape[0]
    assert n in (7, 8, 32) and kps.shape[1] == 3

    if conf is not None:
        w = conf.reshape(-1)
        assert w.shape[0] == n, "conf must have the same length as kps"
    else:
        w = torch.ones(n, dtype=torch.float64, device=kps.device)

    # calculate lengths, widths, and heights from keypoints
    def lw(i, j, scale=1.0):
        d = torch.norm(kps[i] - kps[j]) / scale
        wt = torch.min(w[i], w[j])
        return d * wt, wt

    if n == 8:
        ls_ws = [lw(0, 1), lw(3, 2), lw(4, 5), lw(7, 6)]
        ws_ws = [lw(0, 3), lw(1, 2), lw(4, 7), lw(5, 6)]
        hs_ws = [lw(0, 4), lw(1, 5), lw(2, 6), lw(3, 7)]
    elif n == 7:  # n == 7
        ls_ws = [lw(1, 0, 0.5), lw(2, 0, 0.5), lw(1, 2)]
        ws_ws = [lw(3, 0, 0.5), lw(4, 0, 0.5), lw(3, 4)]
        hs_ws = [lw(5, 0, 0.5), lw(6, 0, 0.5), lw(5, 6)]
    else:  # n == 32
        ls_ws = [
            lw(0, 1), lw(3, 2), lw(4, 5), lw(7, 6),
            lw(16, 18), lw(17, 19), lw(20, 22), lw(21, 23),
            lw(24, 26), lw(25, 27), lw(28, 30), lw(29, 31)]
        ws_ws = [
            lw(0, 3), lw(1, 2), lw(4, 7), lw(5, 6),
            lw(8, 10), lw(9, 11), lw(12, 14), lw(13, 15),
            lw(24, 28), lw(25, 29), lw(26, 30), lw(27, 31)]
        hs_ws = [
            lw(0, 4), lw(1, 5), lw(2, 6), lw(3, 7),
            lw(8, 12), lw(9, 13), lw(11, 15), lw(10, 14),
            lw(16, 20), lw(17, 21), lw(18, 22), lw(19, 23)]

    length = sum(d for d, _ in ls_ws) / (sum(wt for _, wt in ls_ws) + 1e-6)
    width = sum(d for d, _ in ws_ws) / (sum(wt for _, wt in ws_ws) + 1e-6)
    height = sum(d for d, _ in hs_ws) / (sum(wt for _, wt in hs_ws) + 1e-6)

    return [length, width, height]


def get_rotation_axis(kps):
    v1 = kps[0, :] - kps[1, :]
    v2 = kps[3, :] - kps[2, :]
    v3 = kps[4, :] - kps[5, :]
    v4 = kps[7, :] - kps[6, :]
    x_axis = (v1 + v2 + v3 + v4) / 4.0

    v1 = kps[0, :] - kps[3, :]
    v2 = kps[1, :] - kps[2, :]
    v3 = kps[4, :] - kps[7, :]
    v4 = kps[5, :] - kps[6, :]
    y_axis = (v1 + v2 + v3 + v4) / 4.0

    v1 = kps[0, :] - kps[4, :]
    v2 = kps[1, :] - kps[5, :]
    v3 = kps[2, :] - kps[6, :]
    v4 = kps[3, :] - kps[7, :]
    z_axis = (v1 + v2 + v3 + v4) / 4.0

    x_axis = x_axis / torch.norm(x_axis)
    y_axis = y_axis / torch.norm(y_axis)
    z_axis = z_axis / torch.norm(z_axis)

    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    return R


def procrustes_alignment(pts1, pts2, weight=None):
    """
    Procrustes alignment for a single pair of point sets with optional weights.
    Args:
        pts1: (N, 3) tensors.
        pts2: (N, 3) tensors.
        weight: (N,) or (N,1) tensor of non-negative weights.
                If None, equal weights are used.
    Returns:
        R: (3, 3) rotation matrix.
    """
    assert pts1.shape == pts2.shape and pts1.dim() == 2 and pts1.size(1) == 3

    if weight is None:
        weight = torch.ones(
            pts1.shape[0], device=pts1.device, dtype=pts1.dtype)
    weight = weight.view(-1, 1)

    # compute weighted centroids
    W = weight.sum()
    c1 = (weight * pts1).sum(dim=0) / W
    c2 = (weight * pts2).sum(dim=0) / W

    # center the points
    p1 = pts1 - c1
    p2 = pts2 - c2

    # weighted covariance
    H = p1.T @ (weight * p2)

    # SVD and rotation
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.T @ U.T
    if torch.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.T @ U.T

    t = c2 - R @ c1

    return R, t


def rotate_around_x(R, angle_rad):
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
        [0, torch.sin(angle_rad),  torch.cos(angle_rad)]
    ], dtype=R.dtype, device=R.device)
    return R @ Rx


def rotate_around_y(R, angle_rad):
    Ry = torch.tensor([
        [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
        [0, 1, 0],
        [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
    ], dtype=R.dtype, device=R.device)
    return R @ Ry


def rotate_around_z(R, angle_rad):
    Rz = torch.tensor([
        [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
        [torch.sin(angle_rad),  torch.cos(angle_rad), 0],
        [0, 0, 1]
    ], dtype=R.dtype, device=R.device)
    return R @ Rz


def adjust_pose_z_axis_vertical(R_pose, t, K):
    best_R_new = R_pose
    min_diff = float('inf')

    def project_to_image_plane(vec_cam, K):
        vec_cam = vec_cam / vec_cam[2]  # normalize by depth
        proj = K @ vec_cam
        return proj[:2]  # return (x, y)

    for deg in torch.linspace(0, 360, 360):
        angle_rad = torch.deg2rad(deg)
        R_new = rotate_around_x(R_pose, angle_rad)
        vec = R_new[:, 2]  # z-axis
        abs_vec = t.clone()
        abs_vec2 = abs_vec + vec * 0.1

        proj = project_to_image_plane(abs_vec, K)
        proj2 = project_to_image_plane(abs_vec2, K)
        diff = torch.abs(proj[0] - proj2[0])

        if diff < min_diff:
            min_diff = diff
            best_R_new = R_new

    if min_diff > 3:
        print(min_diff)
        print("No rotation found that satisfies condition.")

    return best_R_new


class Box:
    """
    A class representing a 3D bounding box in space.
    3D bounding box corners in canonical order:

        z                    2 -------- 1
        |                   /|         /|
        |                  3 -------- 0 |
        |________ y        | |        | |
       /                   | 6 -------- 5
      /                    |/         |/
     x                     7 -------- 4

    """
    def __init__(self, size, R, t):
        self.size = to_tensor(size, dtype=torch.float64)
        self.R = to_tensor(R, dtype=torch.float64)
        self.t = to_tensor(t, dtype=torch.float64)
        assert self.size.shape == (3,), "size must have three elements"
        assert self.R.shape == (3, 3), "R must be of shape 3x3"
        assert self.t.shape == (3,), "t must have three elements"
        self.canonical_kpts = get_canonical_box(self.size)

        self.flip_pairs = []
        self.symmetric_type = "asymm"

        # indices used for performing interpolation
        self.interp = [
            [0, 1], [3, 2], [4, 5], [7, 6],  # l direction
            [0, 3], [1, 2], [4, 7], [5, 6],  # w direction
            [0, 4], [1, 5], [3, 7], [2, 6]   # h direction
        ]

    def __repr__(self):
        return f"Box(size={self.size}, R={self.R}, t={self.t})"

    def get_pose(self):
        return self.R.clone(), self.t.clone()

    def get_size(self):
        return self.size.clone()

    def update_flip_pairs(self, flip_pairs):
        """
        Update the flip pairs for the box.
        """
        assert isinstance(flip_pairs, (list, np.ndarray)), \
            "flip_pairs must be a list or numpy array"
        self.flip_pairs = flip_pairs

    def update_symmetric_type(self, symmetric_type):
        """
        Update the symmetric type for the box.
        """
        assert isinstance(symmetric_type, str), \
            "symmetric_type must be a string"
        self.symmetric_type = symmetric_type

    def flip_lr(self, intrinsics, baseline, img_width):
        """
        Flip the box left-right based on the camera intrinsics and baseline.
        """
        # ensure intrinsics is 3x3 and baseline is shape (1,)
        assert intrinsics.shape == (3, 3), \
            "intrinsics must be a 3x3 matrix"
        assert baseline.shape == (1,), \
            "baseline must be an array of shape (1,)"

        fx = intrinsics[0, 0]
        cx = intrinsics[0, 2]

        bbox_3d = (self.R @ self.canonical_kpts.T).T + self.t

        # compute x-shift after flipping for box corners
        pts = bbox_3d.clone()
        tx = (img_width - 1 - 2 * cx) * pts[:, 2] / fx
        pts[:, 0] = -pts[:, 0] + baseline[0] + tx

        # compute x-shift after flipping for canonical keypoints
        self.canonical_kpts[:, 0] = -self.canonical_kpts[:, 0]

        pts_ = pts.clone()
        canonical_kpts_ = self.canonical_kpts.clone()
        for pair in self.flip_pairs:
            pts[pair[0]] = pts_[pair[1]]
            pts[pair[1]] = pts_[pair[0]]
            self.canonical_kpts[pair[0]] = canonical_kpts_[pair[1]]
            self.canonical_kpts[pair[1]] = canonical_kpts_[pair[0]]

        self.R, self.t = procrustes_alignment(self.canonical_kpts, pts)

    def redefine_orientation(self, intrinsics):
        stype = self.symmetric_type
        R, t = self.R.clone(), self.t.clone()
        pi_tensor = torch.tensor(np.pi, dtype=R.dtype, device=R.device)

        if stype == 'rotational_symm':  # make z-axis always vertical
            R = adjust_pose_z_axis_vertical(
                R, t, intrinsics[:3, :3])
            # make z point towards the camera
            if R[2, 2] > 0:
                R = rotate_around_x(R, pi_tensor)
            # make z point upward (in case z is parallel to the camera plane)
            if R[1, 2] > 0:
                R = rotate_around_x(R, pi_tensor)
        elif stype == 'two_fold_symm':
            # make y-axis always point toward right
            if R[0, 1] < 0:
                R = rotate_around_z(R, pi_tensor)
            # make z point towards the camera
            if R[2, 2] > 0:
                R = rotate_around_y(R, pi_tensor)
        elif stype == 'mirror_symm':
            # make z point towards the camera
            if R[2, 2] > 0:
                R = rotate_around_x(R, pi_tensor)
        elif stype == 'asymm':
            pass
        else:
            raise ValueError(f"Unknown symmetric type: {stype}")

        self.R = R
        self.t = t

    def get_keypoints(self, num_k, canonical=False):
        """
        Return the 3D keypoints of the bounding box.

        num_k: specify which keypoint representation to return.
            If num_k = 8,
                return all 8 corners of the bounding box.
            If num_k = 7,
                return the box center and the 6 center points of the 6 faces.
            if num_k = 32,
                return 32 keypoints with 8 corners and 24 interpolated points.
        canonical: if True, return the canonical keypoints.
            (i.e., the keypoints in its canonical coordinate without pose).

        Returns:
            kpts3d: (num_k, 3) tensor of keypoints in 3D
        """
        assert num_k in (7, 8, 32), "num_k must be one of (7, 8, 32)"

        # compute 3D corners in world/camera coords
        if canonical:
            kpts3d = self.canonical_kpts.clone()
        else:
            kpts3d = (self.R @ self.canonical_kpts.T).T + self.t

        # if requested, compute the 6 face centers
        if num_k == 7:
            faces = [
                [0, 1, 2, 3, 4, 5, 6, 7],  # center
                [0, 3, 4, 7],  # front
                [1, 2, 5, 6],  # back
                [0, 1, 4, 5],  # left
                [2, 3, 6, 7],  # right
                [0, 1, 2, 3],  # top
                [4, 5, 6, 7],  # bottom
            ]
            kpts3d = torch.stack([kpts3d[face].mean(dim=0) for face in faces])
        elif num_k == 32:
            kpts3d = interpolate(kpts3d, self.interp, [0.332, 0.667])

        return kpts3d

    @classmethod
    def from_keypoints(cls, kps, conf=None, flip_pairs=[], symmetric_type="n"):
        """
        Create a Box object from keypoints.
        kps: (num_k, 3) tensor of keypoints in 3D
        flip_pairs: list of pairs of indices to flip
        symmetric_type: symmetry type
        """
        assert kps.shape[0] in (7, 8, 32), \
            "Number of keypoints must be one of (7, 8, 32)"
        assert kps.shape[1] == 3, "Keypoints must have shape (num_k, 3)"

        # convert numpy inputs to torch tensors
        if not torch.is_tensor(kps):
            kps = to_tensor(kps, dtype=torch.float32)
        if conf is not None:
            if not torch.is_tensor(conf):
                conf = to_tensor(conf, dtype=torch.float32)
            # use 0.5 as threshold
            conf = (conf > 0.5).float()
            # avoid SVD issues with all-zero confidence
            conf = None if torch.sum(conf) < 5 else conf

        size = decode_kpts_to_box(kps, conf)
        R = torch.eye(3, dtype=kps.dtype)
        t = torch.zeros(3, dtype=kps.dtype)

        canonical_box = cls(size, R, t)
        canonical_kpts = canonical_box.get_keypoints(kps.shape[0])
        canonical_kpts = canonical_kpts.to(kps.device, dtype=kps.dtype)

        R, t = procrustes_alignment(canonical_kpts, kps, conf)
        box = cls(size, R, t)
        box.update_flip_pairs(flip_pairs)
        box.update_symmetric_type(symmetric_type)

        return box
