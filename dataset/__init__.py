import torch

from .base import BaseDataset
from utils.util import NestedTensor, get_min_vol_ellipse


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


@torch.jit.unused
def pad_and_resize_tensors(image_list, target_list):
    """support different-sized images"""
    max_size = _max_by_axis([list(img["left"].shape) for img in image_list])
    max_size[-2:] = [(size + 31) // 32 * 32 for size in max_size[-2:]]

    padded_imgs_l, padded_imgs_r = [], []
    padded_masks = []
    resized_targets = {}
    for img, target in zip(image_list, target_list):
        padding = [(s1 - s2)
                   for s1, s2 in zip(max_size, tuple(img["left"].shape))]

        padded_img_l = torch.nn.functional.pad(
            img["left"], (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs_l.append(padded_img_l)

        padded_img_r = torch.nn.functional.pad(
            img["right"], (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs_r.append(padded_img_r)

        m = torch.zeros_like(
            img["left"][0], dtype=torch.int, device=img["left"].device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

        _, ori_h, ori_w = img["left"].shape
        _, new_h, new_w = padded_img_l.shape

        ratio_width = ori_w / new_w
        ratio_height = ori_h / new_h

        target["kpts"] *= torch.as_tensor(
            [ratio_width, ratio_height, 1], dtype=torch.float32)
        target["disp"] *= torch.as_tensor(
            [ratio_width], dtype=torch.float32)

        # area for oks calculation
        kpts = target["kpts"]
        if kpts.shape[0] == 0:
            area = torch.tensor([], dtype=torch.float32)
        else:
            _, radii, _ = get_min_vol_ellipse(kpts[..., :2])
            area = radii[:, 0] * radii[:, 1] * torch.pi
        target["area"] = area

        if "ori_shape" not in resized_targets:
            resized_targets["ori_shape"] = [(new_h, new_w)]
        else:
            resized_targets["ori_shape"].append((new_h, new_w))

        for key in target.keys():
            if key not in resized_targets:
                resized_targets[key] = []

            resized_targets[key].append(target[key])

    images_l = torch.stack(padded_imgs_l)
    images_r = torch.stack(padded_imgs_r)
    masks = torch.stack(padded_masks)

    tensors = NestedTensor(images_l, images_r, masks)

    return tensors, resized_targets


def collate_fn(batch_list):
    image_list, target_list = [], []
    for batch in batch_list:
        image, target = batch
        image_list.append(image)
        target_list.append(target)

    return pad_and_resize_tensors(image_list, target_list)


def build_dataset(cfg, image_set):
    assert image_set in ["train", "val"], \
        f"Invalid image_set: {image_set}. Must be train or val."
    dataset = BaseDataset(cfg, image_set)
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(image_set == "train"),
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
