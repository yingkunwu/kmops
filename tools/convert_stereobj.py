import os
import pickle
import json
import glob
import tqdm
import numpy as np


class StereoObjDataset:
    def __init__(self, root_path):
        self.label_class_mapping = {
            "blade_razor": 0,
            "hammer": 1,
            "needle_nose_pliers": 2,
            "screwdriver": 3,
            "side_cutters": 4,
            "tape_measure": 5,
            "wire_stripper": 6,
            "wrench": 7,
            "centrifuge_tube": 8,
            "microplate": 9,
            "tube_rack": 10,
            "pipette": 11,
            "sterile_tip_rack": 12,
        }

        self.root_path = root_path

    def load_cam_params(self):
        # load camera parameters
        cam_param_filename = os.path.join(self.root_path, 'camera.json')
        with open(cam_param_filename, 'r') as f:
            cam_param = json.load(f)

        proj_matrix_l = cam_param['left']['P']
        proj_matrix_r = cam_param['right']['P']
        baseline = abs(proj_matrix_r[0][3] / proj_matrix_r[0][0])

        return proj_matrix_l, proj_matrix_r, baseline

    def load_annotations(self, subdir, img_id):
        rt_path = os.path.join(
            self.root_path,
            "images_annotations",
            subdir, img_id + '_rt_label.json')
        with open(rt_path, 'r') as f:
            rt_data = json.load(f)

        annotations = []

        for obj in rt_data['class']:
            obj_name = rt_data['class'][obj]

            if obj_name == "centrifuge_tube":
                bbox_filename = os.path.join(
                    self.root_path, 'objects', obj_name + '.bbox')
                with open(bbox_filename, 'r') as f:
                    bbox = f.read().split()
                    bbox = np.array([float(b) for b in bbox])
                    bbox = np.reshape(bbox, (3, 2)).T
                    x_max, x_min = bbox[:, 0]
                    y_max, y_min = bbox[:, 1]
                    z_max, z_min = bbox[:, 2]
            else:
                kp_filename = os.path.join(
                    self.root_path, 'objects', obj_name + '.kp')
                with open(kp_filename, 'r') as f:
                    kps = f.read().split()
                    kps = np.array([float(k) for k in kps])
                    kps = np.reshape(kps, [-1, 3])

                    x_min, x_max = kps[:, 0].min(), kps[:, 0].max()
                    y_min, y_max = kps[:, 1].min(), kps[:, 1].max()
                    z_min, z_max = kps[:, 2].min(), kps[:, 2].max()

            length = x_max - x_min
            width = y_max - y_min
            height = z_max - z_min

            rt = rt_data['rt'][obj]
            R = np.array(rt['R'])
            t = np.array(rt['t'])

            # Ensure length is the longest, followed by width, then height
            dimensions = np.abs(np.array([length, width, height]))
            sorted_indices = np.argsort(dimensions)[::-1]
            length, width, height = dimensions[sorted_indices]
            R = R[:, sorted_indices]  # Reorder the rotation matrix accordingly

            # Re-orthonormalize R to ensure it remains a valid rotation matrix
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R[:, -1] *= -1
            # Check if the rotation matrix is valid
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
                raise ValueError("Rotation matrix is not orthogonal")

            size = [length, width, height]

            found_key = None
            for key in self.label_class_mapping:
                if key in obj_name:
                    found_key = key
                    break

            if not found_key:
                raise KeyError(f"Object name '{obj_name}' "
                               "does not match any key in label_class_mapping")

            annotations.append({
                "obj_name": found_key,
                "label": self.label_class_mapping[found_key],
                "rotation": R.tolist(),
                "translation": t.tolist(),
                "size": size,
            })

        return annotations


def convert(root_path, image_set):
    dataset = StereoObjDataset(root_path)

    if image_set == "train":
        split_filenames = glob.glob(os.path.join(
            root_path, 'split', 'train_*.json'))
    elif image_set == "val":
        split_filenames = glob.glob(os.path.join(
            root_path, 'split', 'val_*.json'))
    else:
        raise ValueError("Invalid image set")

    # load image info: dirname and img_id
    filename_dict = {}
    for split_filename in sorted(split_filenames):
        with open(split_filename, 'r') as f:
            filename_dict.update(json.load(f))
    bad_scenes = [
        "mechanics_scene_7_08162020_5",
        "biolab_scene_2_07312020_7",
        "biolab_scene_2_07312020_11",
        "biolab_scene_2_07312020_13",
    ]
    for scene in bad_scenes:
        filename_dict.pop(scene, None)

    # Load camera parameters
    proj_matrix_l, proj_matrix_r, baseline = dataset.load_cam_params()

    records = []
    for subdir in tqdm.tqdm(filename_dict):
        for img_id in filename_dict[subdir]:
            annots = dataset.load_annotations(subdir, img_id)
            info = {
                'image_id': f"{subdir}_{img_id}",
                'img_path': os.path.join(
                    root_path, "images_annotations", subdir, img_id + '.jpg'),
                'obj_list': annots,
                'proj_matrix_l': proj_matrix_l,
                'proj_matrix_r': proj_matrix_r,
                'baseline': baseline,
            }
            records.append(info)

    os.makedirs("data", exist_ok=True)
    with open(f"data/stereobj_{image_set}.pkl", 'wb') as f:
        pickle.dump(records, f)


if __name__ == "__main__":
    data_root = 'data/stereobj_1m'
    image_sets = ["train", "val"]
    for image_set in image_sets:
        convert(data_root, image_set)
