# KMOPS


This is the official code release for our paper "****". **[[Paper]]  [[Project Page]]**

<video autoplay loop muted playsinline width="960">
    <source src="teaser/biolab_scene_1_07162020_14.gif" type="image/gif" />
</video>

## Table of Contents

- [Quick Demo](#QuickDemo)
- [Dataset Preparation](#Dataset)
- [Train](#Train)
- [Evaluation](#Evaluation)

<a name="QuickDemo"></a>
## Quick Demo

1. Build Docker Image
    ```
    docker build -t kmops docker/
    ```
    > This docker has been tested on:
    > NVIDIA GeForce RTX 4090 / Driver version: 535.230.02 / CUDA version: 12.2
    > NVIDIA H100 PCIe / Driver version: 550.127.08 / CUDA: version 12.4

2. Run Docker container
    ```
    ./run_container.sh
    ```
    > Make sure the ```$WORKSPACE_DIR``` variable in [run_container.sh](run_container.sh) specify the path where you store this repository before running the following command.
    
3. Run inference
    ```
    python demo.py
    ```
    > This code is self-contained and will automatically download the model weights on first run. If the download fails, you can manually download the weights from [here](https://drive.google.com/file/d/1FpzqUp_yxRkaEcyZcvCoNKKgHG0U7dUt/view?usp=sharing) and then run the script again.

<a name="Dataset"></a>
## Dataset Preparation

1. We utilized [StereOBJ-1M](https://sites.google.com/view/stereobj-1m/home?authuser=0) and [Keypose](https://sites.google.com/view/keypose/) in our experiments. Please follow their instructions and prepare the structure as follows:

    * StereOBJ-1M
        ```
        data/stereobj_1m/
        ├── images_annotations/         ← scene image folders
        │   ├── biolab_scene_10_08212020_1/
        │   │   ├── 000000.jpg
        │   │   └── …
        │   ├── biolab_scene_10_08212020_2/
        │   │   ├── 000000.jpg
        │   │   └── …
        │   ├── biolab_scene_10_08212020_3/
        │   │   ├── 000000.jpg
        │   │   └── …
        │   └── …
        ├── objects/                    ← per-object bbox files
        │   ├── blade_razor.bbox
        │   └── …
        ├── split/                      ← train/val/test splits
        │   ├── biolab_object_list.txt
        │   └── …
        ├── camera.json                 ← camera intrinsics/extrinsics
        └── val_label_merged.json       ← merged validation labels
        ```

    * TOD
        ```
        data/tod/
        ├── objects                ← folders that save object mesh files.
        ├── bottle_0/
        │   ├── texture_0_pose_0/  ← folders that save images and annotations
        │   ├── texture_0_pose_1/
        │   ├── texture_0_pose_2/
        │   └── …
        ├── mug_0/
        │   ├── texture_0_pose_0/  ← folders that save images and annotations
        │   ├── texture_0_pose_1/
        │   ├── texture_0_pose_2/
        │   └── …
        ├── cup_0/
        │   ├── texture_0_pose_0/  ← folders that save images and annotations
        │   ├── texture_0_pose_1/
        │   ├── texture_0_pose_2/
        │   └── …
        └── …
        ```
        
2. Convert data into .pkl files
    ```
    python tools/convert_stereobj.py
    ```
    ```
    python tools/convert_keypose.py
    ```
    
    For more datails on how the data in pkl files are organized and how to prepare for custom dataset, please refer to [dataset/custom_dataset.md](dataset/custom_dataset.md).

<a name="Train"></a>
## Training

We use the Hydra library to manage configurations. For more information, please refer to the [Hydra documentation](https://hydra.cc/docs/intro/). Training configs are stored in [conf/](conf/)

```
python train.py --config-name train_stereobj wandb_online=False run_name=8keypoints_stereobj_tempt1
```

<a name="Evaluation"></a>
## Evaluation

Everything required for evaluation will be automatically saved in the folder that saves wandb logging data during training to ensure evaluation uses exactly the same settings as training.
```
python evaluate.py --wandb_folder 8keypoints_o2o_stereobj
```



## Acknowledgments

Many parts of the code are adapted and modified from [DETR](https://github.com/facebookresearch/detr), [RTDETR](https://github.com/lyuwenyu/RT-DETR), [GroupPose](https://github.com/Michel-liu/GroupPose), [Ultralytics](https://github.com/ultralytics/ultralytics), and [SPD](https://github.com/mentian/object-deformnet).

## Citation

If you use this code or the DiverPose dataset for your research, please cite:
