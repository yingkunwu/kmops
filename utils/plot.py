import cv2
import numpy as np


def plot_joints(img, joints, visibility=None, c=(0, 255, 0)):
    """
    joints: Nx2 or Nx3 array where each row is (x, y[, confidence])
    visibility: optional N-element boolean mask
    c: default circle/text color
    """
    joints = np.asarray(joints)
    num_joints, dims = joints.shape

    for k in range(num_joints):
        if dims > 2:
            x, y, conf = joints[k]
        else:
            x, y = joints[k]
            conf = None

        # choose color based on visibility
        color = c
        if visibility is not None and not visibility[k]:
            color = (0, 0, 255)

        # draw the joint
        cv2.circle(img, (int(x), int(y)), 2, color, -1)

        # label the joint index
        cv2.putText(
            img, str(k), (int(x) + 2, int(y) - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

        # if confidence is provided, draw it
        if conf is not None:
            conf_text = f"{conf:.2f}"
            cv2.putText(
                img, conf_text, (int(x) + 2, int(y) + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    return img


def plot_3d_bbox(image, qs, thickness=1, color=[255, 255, 102]):
    """
    Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:

            z                    2 -------- 1
            |                   /|         /|
            |                  3 -------- 0 |
            |________ y        | |        | |
           /                   | 6 -------- 5
          /                    |/         |/
         x                     7 -------- 4
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)

    return image
