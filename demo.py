import os
import tqdm
import glob
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision.transforms.functional as F
import subprocess
import sys
import zipfile

from kmops import build_model
from utils.util import NestedTensor
from utils.vis import visualize

torch.set_float32_matmul_precision("high")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TARGET_DIR = "8keypoints_o2o_stereobj"
FILE_ID = "1I8Zo8qardnml8EGhLZwWL3HqjzJITfrW"
ZIP_PATH = TARGET_DIR + ".zip"


def ensure_gdown():
    try:
        import gdown  # type: ignore
        return gdown
    except Exception:
        pass
    try:
        print("[info] Installing gdown ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"],
            check=True
        )
        import gdown  # type: ignore
        return gdown
    except Exception as e:
        print(f"[warn] gdown unavailable ({e}).")
        return None


def download_with_gdown(file_id: str, out_path: str) -> None:
    gdown = ensure_gdown()
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[info] Downloading via gdown -> {out_path}")
    ok = gdown.download(url, out_path, quiet=False)
    if not ok:
        raise RuntimeError("Not able to download model weight from gdown,"
                           " please download it manually from Google Drive.")


def extract_zip(zip_path: str, expected_dir: str) -> None:
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"{zip_path} is not a valid zip file.")
    with zipfile.ZipFile(zip_path) as zf:
        print(f"[info] Extracting {zip_path} ...")
        zf.extractall(".")
        # If expected dir still not there, just inform user what got extracted
        if not os.path.isdir(expected_dir):
            # Try to guess top-level dirs from the archive
            roots = set(p.split("/")[0] for p in zf.namelist() if "/" in p)
            print(f"[warn] '{expected_dir}' not found after extraction.")
            if roots:
                print(f"[info] Top-level entries in zip: {sorted(roots)}")
    print("[info] Done extracting.")


def preprocess(img_l, img_r, target_size):
    padding = [(s1 - s2) for s1, s2 in zip(target_size, tuple(img_l.shape))]

    padded_img_l = torch.nn.functional.pad(
        img_l, (0, padding[2], 0, padding[1], 0, padding[0]))

    padded_img_r = torch.nn.functional.pad(
        img_r, (0, padding[2], 0, padding[1], 0, padding[0]))

    m = torch.zeros_like(img_l[0], dtype=torch.int, device=img_l.device)
    padded_mask = torch.nn.functional.pad(
        m, (0, padding[2], 0, padding[1]), "constant", 1).to(torch.bool)

    return padded_img_l, padded_img_r, padded_mask


if __name__ == "__main__":
    if not os.path.isdir(TARGET_DIR):
        # Make sure we don't leave a half-downloaded zip around
        if os.path.exists(ZIP_PATH):
            print(f"[info] Removing stale file: {ZIP_PATH}")
            try:
                os.remove(ZIP_PATH)
            except OSError:
                pass

        try:
            # Download model weight using gdown
            download_with_gdown(FILE_ID, ZIP_PATH)
            extract_zip(ZIP_PATH, TARGET_DIR)
        finally:
            # Clean up zip to save space
            if os.path.exists(ZIP_PATH):
                try:
                    os.remove(ZIP_PATH)
                    print(f"[info] Removed archive {ZIP_PATH}")
                except OSError:
                    print(f"[warn] Could not remove {ZIP_PATH};"
                          " remove it manually if desired.")

        if os.path.isdir(TARGET_DIR):
            print(f"[ok] Ready: {TARGET_DIR}")
        else:
            print(f"[warn] Finished, but '{TARGET_DIR}' was not found. "
                  "Please check extracted contents.")
    else:
        print(f"[ok] '{TARGET_DIR}' already exists. Nothing to do.")

    cfg = {
        "ckpt_path": "8keypoints_o2o_stereobj/weight/last.ckpt",
        "demo_path": "./demo",
        "save_path": "./demo_vis",
        "dataset": {
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
        "model": {
            "backbone": "resnet50",
            "num_queries": 150,
            "in_channels": [512, 1024, 2048],
            "feat_strides": [8, 16, 32],
            "num_decoder_points": 4,
            "use_encoder_idx": [2],
            "hidden_dim": 256,
            "dropout": 0.0,
            "nheads": 8,
            "dim_feedforward": 1024,
            "enc_layers": 1,
            "dec_layers": 3
        },
        "aux_loss": False,
        "num_k": 8
    }
    cfg = OmegaConf.create(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.save_path, exist_ok=True)

    PL = torch.tensor([
        [626.1488, 0.0000, 316.1846, 0.0000],
        [0.0000, 626.1488, 349.8279, 0.0000],
        [0.0000, 0.0000, 1.0000, 0.0000]], dtype=torch.float32)
    PR = torch.tensor([
        [626.1488, 0.0000, 316.1846, -28.1518],
        [0.0000, 626.1488, 349.8279, 0.0000],
        [0.0000, 0.0000,   1.0000, 0.0000]], dtype=torch.float32)
    baseline = torch.tensor([0.0450], dtype=torch.float32)

    KMOPS = build_model(cfg)
    ckpt = torch.load(cfg.ckpt_path, weights_only=False)['state_dict']
    KMOPS.load_state_dict(
        {k.replace('model.', ''): v for k, v in ckpt.items()}, strict=True)
    KMOPS = KMOPS.eval().to(device)

    image_paths = glob.glob(os.path.join(cfg.demo_path, "*.jpg"))
    for img_path in tqdm.tqdm(image_paths, desc="Processing images"):
        img = Image.open(img_path)
        width, height = img.size
        img_l = img.crop((0, 0, width // 2, height))
        img_r = img.crop((width // 2, 0, width, height))

        img_l = img_l.resize((640, 640), Image.BILINEAR)
        img_r = img_r.resize((640, 640), Image.BILINEAR)

        # Convert PIL images from RGB to BGR for OpenCV compatibility
        img_l_ori = np.array(img_l)[:, :, ::-1].copy()
        img_r_ori = np.array(img_r)[:, :, ::-1].copy()

        img_l = F.to_tensor(img_l)
        img_l = F.normalize(img_l, mean=MEAN, std=STD)
        img_r = F.to_tensor(img_r)
        img_r = F.normalize(img_r, mean=MEAN, std=STD)

        # Preprocess the image
        img_l, img_r, mask = preprocess(img_l, img_r, (3, 640, 640))
        samples = NestedTensor(img_l, img_r, mask)

        # Run inference
        with torch.no_grad():
            outputs = KMOPS(samples.unsqueeze(0).to(device))
        outputs = KMOPS.postprocess(outputs, [(640, 640)])

        scores = outputs['score'][0]
        labels = outputs['label'][0]
        k2d_l = outputs['kpts_l'][0]
        k2d_r = outputs['kpts_r'][0]
        conf = outputs['conf'][0]
        k3d = KMOPS.reprojection(PL, PR, baseline, k2d_l, k2d_r)
        preds = KMOPS.posefitting(k3d, conf)

        image_name = os.path.splitext(os.path.basename(img_path))
        image_name = f"{image_name[0]}_demo{image_name[1]}"

        visualize(
            img_l_ori, img_r_ori,
            preds['box3ds'], scores, labels,
            preds['ax3ds'], k2d_l, k2d_r,
            cfg.dataset.names, PL, PR,
            os.path.join(cfg.save_path, image_name)
        )
