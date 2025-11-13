import os
import pickle
import glob
import numpy as np
from pathlib import Path


class KITTIDataset:
    def __init__(self, root_path):
        self.label_class_mapping = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
            'Van': 3
        }

        self.root_path = Path(root_path)

    def get_calib(self, sample_idx):
        # load camera parameters
        calib_file = self.root_path / 'calib' / ('%s.txt' % sample_idx)
        assert calib_file.exists()
        with open(calib_file) as f:
            lines = f.readlines()

            obj = lines[2].strip().split(' ')[1:]
            P2 = np.array(obj, dtype=np.float32).reshape(3, 4)
            obj = lines[3].strip().split(' ')[1:]
            P3 = np.array(obj, dtype=np.float32).reshape(3, 4)

        baseline = -(P3[0, 3] - P2[0, 3]) / P2[0, 0]

        return P2.tolist(), P3.tolist(), baseline

    def get_images(self, sample_idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_paths = self.root_path / 'image_02' / ('%s' % sample_idx)
        img_files = sorted(glob.glob(str(img_paths / "*.png")))
        assert len(img_files) > 0
        return img_files

    def get_labels(self, sample_idx):
        label_file = self.root_path / 'label_02' / ('%s.txt' % sample_idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        annotations = {}
        for line in lines:
            label = line.strip().split(' ')
            scene = int(label[0])
            cls_name = label[2]
            if cls_name not in self.label_class_mapping:
                continue
            # l, w, h
            dimension = [float(label[12]), float(label[10]), float(label[11])]
            loc = [float(label[13]), float(label[14]), float(label[15])]
            # compensate y
            loc[1] -= dimension[1] / 2
            ry = float(label[16])
            R = np.array([[np.cos(ry), 0, np.sin(ry)],
                         [0, 1, 0],
                         [-np.sin(ry), 0, np.cos(ry)]])

            if scene in annotations:
                annotations[scene].append({
                    "obj_name": cls_name,
                    "label": self.label_class_mapping[cls_name],
                    "rotation": R.tolist(),
                    "translation": loc,
                    "size": dimension,
                })
            else:
                annotations[scene] = [{
                    "obj_name": cls_name,
                    "label": self.label_class_mapping[cls_name],
                    "rotation": R.tolist(),
                    "translation": loc,
                    "size": dimension,
                }]

        return annotations

    def process_single_scene(self, sample_idx):
        proj_matrix_l, proj_matrix_r, baseline = self.get_calib(sample_idx)
        img_paths = self.get_images(sample_idx)
        labels = self.get_labels(sample_idx)

        infos = []
        for img_path in img_paths:
            image_idx = int(img_path.split('/')[-1].split('.')[0])
            annots = labels.get(image_idx, [])

            info = {
                'image_id': f"{sample_idx}_{image_idx}",
                'img_path': [img_path,
                             img_path.replace('image_02', 'image_03')],
                'obj_list': annots,
                'proj_matrix_l': proj_matrix_l,
                'proj_matrix_r': proj_matrix_r,
                'baseline': baseline,
            }
            infos.append(info)

        return infos


def convert(root_path, image_set):
    dataset = KITTIDataset(root_path)

    if image_set == "train":
        samples = ['0000', '0001', '0002', '0003', '0004', '0005', '0006',
                   '0007', '0008', '0009', '0010', '0011', '0012', '0013',
                   '0014', '0015', '0016', '0017', '0018']
    else:
        samples = ['0019', '0020']

    records = []
    for sample_index in samples:
        infos = dataset.process_single_scene(sample_index)
        records.extend(infos)

    os.makedirs("data", exist_ok=True)
    with open(f"data/kitti_{image_set}.pkl", 'wb') as f:
        pickle.dump(records, f)


if __name__ == "__main__":
    data_root = 'data/kitti/training'
    image_sets = ["train", "val"]
    for image_set in image_sets:
        convert(data_root, image_set)
