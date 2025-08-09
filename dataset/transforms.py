"""
Transforms and data augmentation for both image + keypoints.
"""
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.util import project_3d_to_2d_batch


def make_transforms(cfg, image_set):
    if image_set == 'train':
        return Compose([
            HorizontalFlip(flip_prob=cfg.augments.flip),
            RandomSizeCrop(*cfg.augments.random_size_crop),
            ScaleJitter(target_size=cfg.augments.resize,
                        scale_range=cfg.augments.scale_jitter),
            Collect(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    num_k=cfg.num_k),
        ])
    elif image_set in ['val', 'test']:
        return Compose([
            ScaleJitter(target_size=cfg.augments.resize,
                        scale_range=[1.0, 1.0]),
            Collect(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    num_k=cfg.num_k),
        ])

    else:
        raise ValueError(f'unknown {image_set}')


class HorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            for key in ["left", "right"]:
                image[key] = F.hflip(image[key])

            # Swap left and right views
            image["left"], image["right"] = \
                image["right"], image["left"].copy()

            if target is None:
                return image, None

            _, _, width = F.get_dimensions(image["left"])
            intrinsics = target["proj_matrix_l"][:3, :3]
            baseline = target["baseline"]
            for box in target["boxes"]:
                box.flip_lr(intrinsics, baseline, width)

        return image, target


class RandomSizeCrop(object):
    def __init__(self, min_ratio, max_ratio):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, image, target):
        _, orig_height, orig_width = F.get_dimensions(image["left"])
        min_width = min(orig_width, self.max_ratio * orig_width)
        min_height = min(orig_height, self.max_ratio * orig_height)

        if self.min_ratio * orig_width > min_width \
                or self.min_ratio * orig_height > min_height:
            return image, target

        w = random.randint(round(self.min_ratio * orig_width), min_width)
        h = random.randint(round(self.min_ratio * orig_height), min_height)

        region = T.RandomCrop.get_params(image['left'], [h, w])
        image['left'] = F.crop(image['left'], *region)
        image['right'] = F.crop(image['right'], *region)

        if target is None:
            return image, None

        i, j, h, w = region

        # Compute the transformation matrix
        transform_matrix = torch.eye(3).to(torch.float64)
        transform_matrix[0, 2] = -j
        transform_matrix[1, 2] = -i

        # Update projection matrices
        for key in ["proj_matrix_l", "proj_matrix_r"]:
            proj_matrix = target[key]
            target[key] = torch.mm(transform_matrix, proj_matrix[:3])

        return image, target


class ScaleJitter(object):
    """
    Randomly resizes the image and its keypoints within the specified scale
    range.

    Args:
        target_size (tuple of ints): The target size for the transform
            provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b),
            then scale is randomly sampled from the range a <= scale <= b.
    """

    def __init__(self, target_size, scale_range=(0.1, 2.0)):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range

    def __call__(self, image, target=None):
        _, orig_height, orig_width = F.get_dimensions(image["left"])

        scale = self.scale_range[0] + torch.rand(1) \
            * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height,
                self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image["left"] = F.resize(image["left"], [new_height, new_width])
        image["right"] = F.resize(image["right"], [new_height, new_width])

        ratio_width = new_width / orig_width
        ratio_height = new_height / orig_height

        for key in ["proj_matrix_l", "proj_matrix_r"]:
            proj_matrix = target[key]
            proj_matrix[0, 0] *= ratio_width
            proj_matrix[1, 1] *= ratio_height
            proj_matrix[0, 2] *= ratio_width
            proj_matrix[1, 2] *= ratio_height
            if key == "proj_matrix_r":
                proj_matrix[0, -1] *= ratio_width
            target[key] = proj_matrix

        return image, target


class Collect(object):
    def __init__(self, mean, std, num_k):
        self.mean = mean
        self.std = std
        self.num_k = num_k

    def __call__(self, image, target=None):
        for key in ["left", "right"]:
            image[key] = F.to_tensor(image[key])
            image[key] = F.normalize(image[key], mean=self.mean, std=self.std)

        if target is None:
            return image, None

        h, w = image["left"].shape[-2:]
        intrinsics = target["proj_matrix_l"][:3, :3]
        proj_matrix_l = target["proj_matrix_l"]
        proj_matrix_r = target["proj_matrix_r"]

        target["kpts_3d"], target["kpts_canonical"] = [], []
        size_list, R_list, t_list = [], [], []
        for box in target["boxes"]:
            box.redefine_orientation(intrinsics)
            target["kpts_3d"].append(box.get_keypoints(self.num_k))
            target["kpts_canonical"].append(
                box.get_keypoints(self.num_k, canonical=True))
            R, t = box.get_pose()
            size = box.get_size()
            size_list.append(size)
            R_list.append(R)
            t_list.append(t)
        target["kpts_3d"] = torch.stack(target["kpts_3d"], dim=0)
        target["kpts_canonical"] = torch.stack(target["kpts_canonical"], dim=0)
        target["size"] = torch.stack(size_list, dim=0)
        target["R"] = torch.stack(R_list, dim=0)
        target["t"] = torch.stack(t_list, dim=0)

        kpts3d = target["kpts_3d"].clone()
        kpts2d_l = project_3d_to_2d_batch(kpts3d, proj_matrix_l)
        kpts2d_r = project_3d_to_2d_batch(kpts3d, proj_matrix_r)

        visibility = torch.ones_like(kpts3d[:, :, :1])
        kpts2d_l = torch.cat([kpts2d_l, visibility], dim=-1)
        kpts2d_r = torch.cat([kpts2d_r, visibility], dim=-1)

        for kpts2d in [kpts2d_l, kpts2d_r]:
            V = kpts2d[:, :, 2:]  # (num_object, num_keypoints, 1)
            Z = kpts2d[:, :, :2]  # (num_object, num_keypoints, 2)
            Z[:, :, 0] = Z[:, :, 0] / w
            Z[:, :, 1] = Z[:, :, 1] / h

            invalid = (Z[:, :, 0] < 0) | (Z[:, :, 0] > 1) \
                | (Z[:, :, 1] < 0) | (Z[:, :, 1] > 1)
            V[invalid] = 0
            kpts2d = torch.cat([Z, V], dim=2)

        kpts_mask = torch.minimum(kpts2d_l[..., 2:], kpts2d_r[..., 2:])
        kpts2d_mid = (kpts2d_l + kpts2d_r).mul(0.5)
        kpts2d_mid[..., 2:] = kpts_mask
        target["kpts"] = kpts2d_mid
        target["disp"] = kpts2d_l[..., :1] - kpts2d_r[..., :1]

        # Remove objects that are not visible in both views
        visible = (kpts2d_l[:, :, 2] > 0) & (kpts2d_r[:, :, 2] > 0)
        num_visible_pts = torch.sum(visible, dim=1)

        keep = num_visible_pts > 3
        for key in ["kpts", "disp", "kpts_3d", "labels",
                    "kpts_canonical", "size", "R", "t"]:
            target[key] = target[key][keep]

        # ensure floating type is float32
        for key in ["kpts", "disp", "kpts_3d", "kpts_canonical",
                    "size", "R", "t"]:
            target[key] = target[key].to(torch.float32)

        # won't be used after this
        del target["boxes"]

        return image, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
