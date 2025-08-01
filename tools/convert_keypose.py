import glob
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

# ensure parent directory is on the module search path
parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
sys.path.insert(0, parent_dir)

from keypose import utils  # noqa: E402
from utils.plot import plot_3d_bbox  # noqa: E402


def get_box_3d(box, R, t):
    """
    Get the 3D bounding box corners from the 3D bounding box parameters.
    Return:
        (8,3) array of vertices for the 3d box in the following order:

            z                    2 -------- 1
            |                   /|         /|
            |                  3 -------- 0 |
            |________ y        | |        | |
           /                   | 6 -------- 5
          /                    |/         |/
         x                     7 -------- 4
    """
    x_min, x_max = box[0]
    y_min, y_max = box[1]
    z_min, z_max = box[2]

    length = x_max - x_min
    width = y_max - y_min
    height = z_max - z_min

    # 3d bounding box corners
    x_corners = [length / 2, -length / 2, -length / 2, length / 2,
                 length / 2, -length / 2, -length / 2, length / 2]
    y_corners = [width / 2, width / 2, -width / 2, -width / 2,
                 width / 2, width / 2, -width / 2, -width / 2]
    z_corners = [height / 2, height / 2, height / 2, height / 2,
                 -height / 2, -height / 2, -height / 2, -height / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners])).T + t
    return corners_3d


def draw_circle(image, uvd, color, size=3):
    # Filled color circle.
    cv2.circle(image, (int(uvd[0]), int(uvd[1])), size, color, -1)
    # White outline.
    cv2.circle(image, (int(uvd[0]), int(uvd[1])), size + 1, (255, 255, 255))


def adjust_pose(R, box, adjust=""):
    def rotate_around_x(R, angle_rad):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad),  np.cos(angle_rad)]
        ], dtype=R.dtype)
        return R @ Rx

    def rotate_around_y(R, angle_rad):
        Ry = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ], dtype=R.dtype)
        return R @ Ry

    def rotate_around_z(R, angle_rad):
        Rz = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0, 0, 1]
        ], dtype=R.dtype)
        return R @ Rz

    for axis in adjust.split(" -> "):
        if axis == "x":
            R = rotate_around_x(R, -np.pi / 2)
        elif axis == "y":
            R = rotate_around_y(R, -np.pi / 2)
        elif axis == "z":
            R = rotate_around_z(R, -np.pi / 2)

        # Adjust the box dimensions based on the final rotation
        if axis == "x":
            box[1], box[2] = box[2].copy(), box[1].copy()  # Swap y and z
        if axis == "y":
            box[0], box[2] = box[2].copy(), box[0].copy()  # Swap x and z
        if axis == "z":
            box[0], box[1] = box[1].copy(), box[0].copy()  # Swap x and y

    return R, box


def show_kps(im_l, im_r, kps, obj, cat):
    """Draw left/right images and keypoints using OpenCV."""
    cam, uvds, transform = kps
    baseline = cam.baseline

    im_l = cv2.cvtColor(im_l, cv2.COLOR_BGR2RGB)
    im_r = cv2.cvtColor(im_r, cv2.COLOR_BGR2RGB)

    colors = 255 * plt.get_cmap('rainbow')(np.linspace(0, 1.0, 10))[:, :3]
    uvds = np.array(uvds)
    for i, uvd in enumerate(uvds):
        draw_circle(im_l, uvd, colors[i * 3], 3)

    p_matrix = utils.p_matrix_from_camera(cam)
    q_matrix = utils.q_matrix_from_camera(cam)
    xyzs = utils.project_np(q_matrix, uvds.T)

    transform, uvds = obj.project_to_uvd(xyzs, p_matrix)
    xyzs = utils.project_np(q_matrix, uvds.T).T
    R = transform[:3, :3]

    vertices = obj.vertices
    box = [
        [np.min(vertices[:, 0]), np.max(vertices[:, 0])],
        [np.min(vertices[:, 1]), np.max(vertices[:, 1])],
        [np.min(vertices[:, 2]), np.max(vertices[:, 2])]
    ]

    if cat in ["mug_2", "heart_0"]:
        adjust = "x -> z"
    elif cat == "ball_0":
        adjust = "x -> y -> y"
    elif cat in ["mug_3", "mug_5"]:
        adjust = "z"
    elif cat in ["bottle_1", "bottle_2"]:
        adjust = "x"
    elif cat == "tree_0":
        adjust = "x -> x"
    else:
        adjust = ""
    R, box = adjust_pose(R, box, adjust)

    # switch z and x axis
    adjust = "y -> x -> x"
    R, box = adjust_pose(R, box, adjust)

    xmin, xmax = np.min(xyzs[:, 0]), np.max(xyzs[:, 0])
    ymin, ymax = np.min(xyzs[:, 1]), np.max(xyzs[:, 1])
    zmin, zmax = np.min(xyzs[:, 2]), np.max(xyzs[:, 2])
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    t = np.array([cx, cy, cz])

    im_l = obj.draw_points(im_l)

    k_matrix = utils.k_matrix_from_camera(cam)
    p_left = np.eye(4)
    p_left[:3, :3] = k_matrix
    p_left[:3, 3] = np.array([0, 0, 0])
    p_right = np.eye(4)
    p_right[:3, :3] = k_matrix
    p_right[:3, 3] = np.array([-baseline * k_matrix[0, 0], 0, 0])

    box3d = get_box_3d(box, R, t)
    box3d_hom = np.hstack((box3d, np.ones((8, 1))))

    box3dL = (p_left[:3] @ box3d_hom.T).T
    box3dL = box3dL[:, :2] / box3dL[:, 2:]

    box3dR = (p_right[:3] @ box3d_hom.T).T
    box3dR = box3dR[:, :2] / box3dR[:, 2:]

    plot_3d_bbox(im_l, box3dL, 2, [255, 0, 0])
    plot_3d_bbox(im_r, box3dR, 2, [0, 255, 0])

    # Define the pose axis in 3D space
    axis_length = 0.1  # Adjust the length of the axis as needed
    axis_3d = np.array([
        [0, 0, 0],  # Origin
        [axis_length, 0, 0],  # X-axis
        [0, axis_length, 0],  # Y-axis
        [0, 0, axis_length]   # Z-axis
    ])

    # Transform the axis points using R and t
    axis_3d = (R @ axis_3d.T).T + t

    axis_3d_hom = np.hstack((axis_3d, np.ones((4, 1))))

    axis_3d_hom = (p_left @ axis_3d_hom.T).T
    axis_2d = axis_3d_hom[:, :2] / axis_3d_hom[:, 2:3]
    axis_2d = axis_2d.astype(np.int32)

    # Draw the pose axis on the image
    cv2.line(im_l, tuple(axis_2d[0]), tuple(axis_2d[1]), (0, 0, 255), 2)
    cv2.line(im_l, tuple(axis_2d[0]), tuple(axis_2d[2]), (0, 255, 0), 2)
    cv2.line(im_l, tuple(axis_2d[0]), tuple(axis_2d[3]), (255, 0, 0), 2)

    # image = np.hstack((im_l, im_r))
    # cv2.imshow('image', image)
    key = cv2.waitKey(10)

    x_min, x_max = box[0]
    y_min, y_max = box[1]
    z_min, z_max = box[2]

    length = x_max - x_min
    width = y_max - y_min
    height = z_max - z_min

    size = [length, width, height]

    obj_info = {
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "size": size
    }
    cam_info = {
        "p_left": p_left.tolist(),
        "p_right": p_right.tolist(),
        "baseline": baseline
    }

    if key == ord('q'):
        return True, obj_info, cam_info
    return False, obj_info, cam_info


def convert(image_dir, mesh_file, cat):
    print('Looking for images in %s' % image_dir)
    filenames = glob.glob(os.path.join(image_dir, '*_L.png'))
    if not filenames:
        print("Couldn't find any PNG files in %s" % image_dir)
        exit(-1)
    filenames.sort()
    print('Found %d files in %s' % (len(filenames), image_dir))

    obj = None
    if mesh_file:
        obj = utils.read_mesh(mesh_file)

    records = []
    for fname in filenames:
        im_l = utils.read_image(fname)
        im_r = utils.read_image(fname.replace('_L.png', '_R.png'))
        cam, _, _, uvds, _, _, transform = utils.read_contents_pb(
            fname.replace('_L.png', '_L.pbtxt')
        )
        ret, obj_info, cam_info = \
            show_kps(im_l, im_r, (cam, uvds, transform), obj, cat)
        if ret:
            break

        stem = os.path.splitext(os.path.basename(mesh_file))[0][:-2]
        obj_info['obj_name'] = stem
        obj_info['label'] = 0

        subdir = os.path.basename(image_dir)
        img_name = os.path.splitext(
            os.path.basename(fname))[0].replace('_L', '')

        info = {
            'image_id': f"{subdir}_{img_name}",
            'img_path': [fname, fname.replace('_L.png', '_R.png')],
            'obj_list': [obj_info],
            'proj_matrix_l': cam_info['p_left'],
            'proj_matrix_r': cam_info['p_right'],
            'baseline': cam_info['baseline'],
        }
        records.append(info)

    return records


def collect(root_path, category, image_set, set_name):
    records = []

    for cat in category:
        image_dirs = glob.glob(os.path.join(root_path, cat, "*"))
        image_dirs.sort()
        mesh_file = os.path.join(root_path, "objects", cat + ".obj")
        for image_dir in image_dirs:
            rec = convert(image_dir, mesh_file, cat)
            records.extend(rec)

    os.makedirs("data", exist_ok=True)
    with open(f"data/tod_{set_name}_{image_set}.pkl", 'wb') as f:
        pickle.dump(records, f)


def main():
    root_path = "data/tod"

    # Collect training and validation data for the bottle category
    bottle_train = ["bottle_1", "bottle_2"]
    bottle_val = ["bottle_0"]
    collect(root_path, bottle_train, "train", "bottle")
    collect(root_path, bottle_val, "val", "bottle")

    # Collect training and validation data for the bottle and cup category
    bottle_cup_train = ["bottle_1", "bottle_2", "cup_0"]
    bottle_cup_val = ["bottle_0", "cup_1"]
    collect(root_path, bottle_cup_train, "train", "bottle_cup")
    collect(root_path, bottle_cup_val, "val", "bottle_cup")

    # Collect training and validation data for the mug category
    mug_train = ["mug_1", "mug_2", "mug_3", "mug_4", "mug_5", "mug_6"]
    mug_val = ["mug_0"]
    collect(root_path, mug_train, "train", "mug")
    collect(root_path, mug_val, "val", "mug")


if __name__ == '__main__':
    main()
