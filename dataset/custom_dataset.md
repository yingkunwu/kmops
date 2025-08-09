# Customized Dataset

This section describes how to prepare a customized dataset for use with KMOPS. The data preparation pipeline converts your dataset into a standard `.pkl` format, which can be directly used for training or evaluation.

## Coordinate System - Object Pose Definition

We use right-handed coordinate frames for objects and cameras. All objects have a per-category canonically oriented frames. This means that the X, Y and Z axes of all objects in the same category are consistently oriented. The pose of the object represents the transformation from the object coordinate to the camera coordinate.

* Camera coordinate system:
    ```
               up    z (forward)
               |     ↗
               |    /
               |   /
               |  /
               | /
               |/
    left  ---- O ----→ x (right)
               |
               |
               |
               |
               ↓
             y (down)
    ```
    * +X → right
    * +Y → down
    * +Z → forward (view direction)

* Object coordinate system:
    ```
            z (up) ↑     x (forward)
                   |     ↗
                   |    /
                   |   /
                   |  /
                   | /
                   |/
    y (left) ←---- O ----  right
                   |
                   |
                   |
                   |

                 down
    ```
    * +X → forward
    * +Y → left
    * +Z → up

## Expected Output Format (`.pkl`)

Each `.pkl` file should be a list of dictionary samples with the following keys:

```python
{
  "image_id": str,
  "img_path": str,        # Path to the RGB image
  "proj_matrix_l": list,  # Left calibrated camera projection matrix, shape (3, 4)
  "proj_matrix_4": list,  # Right calibrated camera projection matrix, shape (3, 4)
  "baseline": float       # The calibrated stereo camera baseline
  "obj_list": [           # List of 3D object annotations
    {
      "obj_name": str           # Name of the object for classification
      "label": int              # Label of the object for classification
      "rotation": list          # 3x3 rotation matrix that represents the orientation of the object
      "translation": [x, y, z]  # 1x3 vector that represents the location of the object
      "size": [l, w, h]         # 1x3 vector that represents the true scale of the object
    }
  ]
}
```

> Note that the obj_name and label for each object should be in accordance with the dataset.names in [train config](conf/).


## Object Property (Optional)
To enable horizontal flipping during training, you must handle object symmetries correctly. Extend the object properties in [dataset/object_property.py](dataset/object_property.py) to specify each object’s behavior under a horizontal flip. In this augmentation we negate the x coordinate in the object’s local coordinate frame. Please carefully consider how your object’s appearance should change accordingly. The ```flip_pairs``` mapping should define how the eight 3D box corners permute under the flip, using the canonical ordering below.
```
3D bounding box corners in canonical order:

    z                    2 -------- 1
    |                   /|         /|
    |                  3 -------- 0 |
    |________ y        | |        | |
   /                   | 6 -------- 5
  /                    |/         |/
 x                     7 -------- 4
```


## Data Validation (Optional)

You can visually check your 3D box and pose using the following code:

```python
import os
import tqdm
import torch
from omegaconf import OmegaConf

from kmops import KMOPS
from dataset import build_dataset
from utils.util import kpts_disp_to_left_right
from utils.vis import visualize


def get_rotation_axis(kps):
    """
    Compute a 3x3 rotation matrix R from 3D keypoints.
    """
    N = kps.shape[0]
    if N in (8, 32):
        # X axis: average of the 4 edge vectors along X
        v1 = kps[0] - kps[1]
        v2 = kps[3] - kps[2]
        v3 = kps[4] - kps[5]
        v4 = kps[7] - kps[6]
        x_axis = (v1 + v2 + v3 + v4) / 4.0

        # Y axis: average of the 4 edge vectors along Y
        v1 = kps[0] - kps[3]
        v2 = kps[1] - kps[2]
        v3 = kps[4] - kps[7]
        v4 = kps[5] - kps[6]
        y_axis = (v1 + v2 + v3 + v4) / 4.0

        # Z axis: average of the 4 edge vectors along Z
        v1 = kps[0] - kps[4]
        v2 = kps[1] - kps[5]
        v3 = kps[2] - kps[6]
        v4 = kps[3] - kps[7]
        z_axis = (v1 + v2 + v3 + v4) / 4.0
    else:
        # Fallback for other keypoint counts
        x_axis = kps[1] - kps[2]
        y_axis = kps[3] - kps[4]
        z_axis = kps[5] - kps[6]

    # Normalize each axis
    x_axis = x_axis / x_axis.norm(p=2)
    y_axis = y_axis / y_axis.norm(p=2)
    z_axis = z_axis / z_axis.norm(p=2)

    # Stack as columns to form the rotation matrix
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return R


def run(cfg, train_loader):
    # Save to local folder
    output_dir = "dataset_preview"
    os.makedirs(output_dir, exist_ok=True)

    for i, (samples, target) in enumerate(tqdm.tqdm(train_loader)):
        image_l, image_r, mask = samples.decompose()

        for j in range(image_l.shape[0]):
            kpts = target["kpts"][j]
            disp = target["disp"][j]
            labels = target["labels"][j]
            scores = torch.ones_like(labels, dtype=torch.float32)
            ori_shape = target["ori_shape"][j]

            kpts[..., 0] = kpts[..., 0] * ori_shape[1]
            kpts[..., 1] = kpts[..., 1] * ori_shape[0]
            disp[..., 0] = disp[..., 0] * ori_shape[1]

            kpts_l, kpts_r = kpts_disp_to_left_right(kpts, disp)
            kpts_mask = kpts[..., 2:]

            # decode & project bboxes and axes
            PL = target['proj_matrix_l'][j].to(torch.float32)
            PR = target['proj_matrix_r'][j].to(torch.float32)
            baseline = target['baseline'][j].to(torch.float32)

            k3d = KMOPS.reprojection(PL, PR, baseline, kpts_l, kpts_r)
            preds = KMOPS.posefitting(k3d, kpts_mask)

            visualize(
                image_l[j], image_r[j],
                preds['box3ds'], scores, labels,
                preds['ax3ds'], kpts_l, kpts_r,
                cfg.dataset.names, PL, PR,
                os.path.join(output_dir, f"sample_{i}_{j}.png"),
                normalize_image=True
            )


if __name__ == "__main__":
    config_dict = {
        "dataset": {
            "train_pkl": "data/stereobj_train.pkl",
            "names": {
                0: "blade_razor",
                1: "hammer",
                2: "needle_nose_pliers",
                3: "screwdriver",
                4: "side_cutters",
                5: "tape_measure",
                6: "wire_stripper",
                7: "wrench",
                8: "centrifuge_tube",
                9: "microplate",
                10: "tube_rack",
                11: "pipette",
                12: "sterile_tip_rack"
            }
        },
        "num_k": 8,  # you have option for (7, 8, 32)
        "batch_size": 4,
        "num_workers": 0,
        "augments": {
            "flip": 0.5,
            "random_size_crop": [1, 1],
            "scale_jitter": [1, 1],
            "resize": [640, 640]
        },
    }
    cfg = OmegaConf.create(config_dict)
    train_loader = build_dataset(cfg, "train")
    run(cfg, train_loader)

```

> This helps ensure that your pose and 3D geometry are correctly aligned with the projection model.

### Acknowledgments

This data preparation methodology is adapted from DetAny3D.