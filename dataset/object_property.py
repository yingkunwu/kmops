def get_symmetric_type(obj_name):
    if obj_name in ['centrifuge_tube', 'screwdriver']:
        symmetric_type = "rotational_symm"
    elif obj_name in ['microplate', 'tube_rack']:
        symmetric_type = "two_fold_symm"
    elif obj_name in ["needle_nose_pliers", "side_cutters", "wire_stripper"]:
        symmetric_type = "mirror_symm"
    else:
        symmetric_type = "asymm"

    return symmetric_type


def get_flip_pairs(obj_name):
    if obj_name in ['hammer', "wrench", "pipette"]:
        flip_pairs = [[0, 4], [3, 7], [1, 5], [2, 6]]
    elif obj_name in ["sterile_tip_rack", "screwdriver", "centrifuge_tube",
                      "needle_nose_pliers", "side_cutters", "wire_stripper"]:
        flip_pairs = [[0, 3], [4, 7], [5, 6], [1, 2]]
    elif obj_name in ["microplate", "tube_rack"]:
        flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7]]
    else:
        flip_pairs = []

    return flip_pairs
