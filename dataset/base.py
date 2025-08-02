import copy
import pickle
import torch
from PIL import Image

from .transforms import make_transforms
from .object_property import get_symmetric_type, get_flip_pairs
from utils.box import Box
from utils.util import to_tensor


class BaseDataset:
    def __init__(self, cfg, image_set):
        if image_set == "train":
            pkl_path = cfg.dataset.train_pkl
        else:
            pkl_path = cfg.dataset.val_pkl
        with open(pkl_path, 'rb') as f:
            self.pkl = pickle.load(f)

        self.transform = make_transforms(cfg, image_set)

    def _load_image(self, img_path):
        # if img_path is a list or tuple, assume it contains
        # : [left_image_path, right_image_path]
        if isinstance(img_path, (list, tuple)):
            img_l = Image.open(img_path[0])
            img_r = Image.open(img_path[1])
            return img_l, img_r

        # otherwise treat img_path as a single stereo image
        img = Image.open(img_path)
        width, height = img.size
        img_l = img.crop((0, 0, width // 2, height))
        img_r = img.crop((width // 2, 0, width, height))
        return img_l, img_r

    def __len__(self):
        return len(self.pkl)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.pkl[idx])
        # validate that all required keys are present in db_rec
        required_keys = ["img_path", "obj_list", "proj_matrix_l",
                         "proj_matrix_r", "baseline"]
        missing = [k for k in required_keys if k not in db_rec]
        if missing:
            raise KeyError(f"db_rec is missing required keys: {missing}")

        # validate that all required keys are present in items within obj_list
        required_keys = [
            "obj_name", "label", "rotation", "translation", "size"]
        for obj in db_rec["obj_list"]:
            missing = [k for k in required_keys if k not in obj]
            if missing:
                raise KeyError("item within obj_list is missing required keys"
                               f": {missing}")

        img_l, img_r = self._load_image(db_rec['img_path'])
        image = {"left": img_l, "right": img_r}

        boxes, labels = [], []
        for obj in db_rec["obj_list"]:
            rotation = torch.tensor(obj["rotation"], dtype=torch.float32)
            translation = torch.tensor(obj["translation"], dtype=torch.float32)
            size = torch.tensor(obj["size"], dtype=torch.float32)
            box = Box(size, rotation, translation)

            flip_pairs = get_flip_pairs(obj["obj_name"])
            symmetric_type = get_symmetric_type(obj["obj_name"])
            box.update_flip_pairs(flip_pairs)
            box.update_symmetric_type(symmetric_type)
            boxes.append(box)
            labels.append(obj["label"])

        target = {
            "img_path": db_rec["img_path"],
            "labels": to_tensor(labels, dtype=torch.int64),
            "boxes": boxes,
            "proj_matrix_l": to_tensor(
                db_rec['proj_matrix_l'], dtype=torch.float64),
            "proj_matrix_r": to_tensor(
                db_rec['proj_matrix_r'], dtype=torch.float64),
            "baseline": to_tensor(
                [db_rec['baseline']], dtype=torch.float64)
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
