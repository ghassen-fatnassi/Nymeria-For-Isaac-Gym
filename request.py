from nymeria_files.xsens_constants import XSensConstants

print("XSens Part Names:")
for i, name in enumerate(XSensConstants.part_names):
    print(f"{i}: {name}")

def get_proto_skeleton_tree():
    node_names = [
        'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
        'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
        'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
    ]
    return node_names

print("\nProtoMotion Node Names:")
for i, name in enumerate(get_proto_skeleton_tree()):
    print(f"{i}: {name}")