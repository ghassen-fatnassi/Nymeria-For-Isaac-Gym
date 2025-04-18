import numpy as np
import os
from collections import OrderedDict
import argparse
from scipy.spatial.transform import Rotation as R # Import Rotation for SLERP

# Assuming these files exist and are correctly defined
from nymeria_files.xsens_constants import XSensConstants
from nymeria_files.body_motion_provider import create_body_data_provider

# Removed the old create_xsens_to_proto_mapping function

def get_proto_skeleton_tree():
    """
    Defines the target SMPL/ProtoMotion skeleton structure.
    NOTE: Ensure this matches the SMPL definition you intend to use.
    The mapping logic below relies on these names and indices.
    """
    node_names = [
        'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
        'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
        'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
    ]
    # Parent indices for the SMPL/ProtoMotion skeleton
    parent_indices = np.array([
        -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17,  # Torso, Head, Left arm chain
        11, 19, 20, 21, 22                                                # Right arm chain
    ], dtype=np.int64)
    # Local translations (T-pose offsets) for the SMPL/ProtoMotion skeleton
    # You might need to adjust these if your target SMPL model has different offsets
    local_translation_data = np.array([
        [-0.0018, -0.2233,  0.0282],
        [-0.0068,  0.0695, -0.0914], # L_Hip relative to Pelvis
        [-0.0045,  0.0343, -0.3752], # L_Knee relative to L_Hip
        [-0.0437, -0.0136, -0.398 ], # L_Ankle relative to L_Knee
        [ 0.1193,  0.0264, -0.0558], # L_Toe relative to L_Ankle
        [-0.0043, -0.0677, -0.0905], # R_Hip relative to Pelvis
        [-0.0089, -0.0383, -0.3826], # R_Knee relative to R_Hip
        [-0.0423,  0.0158, -0.3984], # R_Ankle relative to R_Knee
        [ 0.1233, -0.0254, -0.0481], # R_Toe relative to R_Ankle
        [-0.0267, -0.0025,  0.109 ], # Torso (Spine1?) relative to Pelvis
        [ 0.0011,  0.0055,  0.1352], # Spine (Spine2?) relative to Torso
        [ 0.0254,  0.0015,  0.0529], # Chest (Spine3?) relative to Spine
        [-0.0429, -0.0028,  0.2139], # Neck relative to Chest
        [ 0.0513,  0.0052,  0.065 ], # Head relative to Neck
        [-0.0341,  0.0788,  0.1217], # L_Thorax (L_Collar?) relative to Chest
        [-0.0089,  0.091 ,  0.0305], # L_Shoulder relative to L_Thorax
        [-0.0275,  0.2596, -0.0128], # L_Elbow relative to L_Shoulder
        [-0.0012,  0.2492,  0.009 ], # L_Wrist relative to L_Elbow
        [-0.0149,  0.084 , -0.0082], # L_Hand relative to L_Wrist
        [-0.0386, -0.0818,  0.1188], # R_Thorax (R_Collar?) relative to Chest
        [-0.0091, -0.096 ,  0.0326], # R_Shoulder relative to R_Thorax
        [-0.0214, -0.2537, -0.0133], # R_Elbow relative to R_Shoulder
        [-0.0056, -0.2553,  0.0078], # R_Wrist relative to R_Elbow
        [-0.0103, -0.0846, -0.0061]  # R_Hand relative to R_Wrist
    ], dtype=np.float32)

    return OrderedDict({
        'node_names': node_names,
        'parent_indices': OrderedDict({'arr': parent_indices, 'context': {'dtype': 'int64'}}),
        'local_translation': OrderedDict({'arr': local_translation_data, 'context': {'dtype': 'float32'}})
    })

def create_proto_motion_from_dataprovider(data_provider):
    if data_provider is None:
        print("Error: Data provider is None.")
        return None

    # --- 1. Get Data and Metadata ---
    try:
        num_frames = data_provider.xsens_data[XSensConstants.k_frame_count][0]
        frame_rate = data_provider.xsens_data[XSensConstants.k_framerate][0]
        # Reshape XSens data: (num_frames, num_nymeria_joints, 4/3)
        # Assuming WXYZ format for quaternions from XSens
        segment_quat_wxyz = data_provider.xsens_data[XSensConstants.k_part_qWXYZ].reshape(num_frames, XSensConstants.num_parts, 4)
        segment_trans = data_provider.xsens_data[XSensConstants.k_part_tXYZ].reshape(num_frames, XSensConstants.num_parts, 3)
        # Access and reshape velocity and angular velocity using string keys
        # Based on the provided output, these keys exist directly in the dictionary
        segment_velocity = data_provider.xsens_data['segment_velocity'].reshape(num_frames, XSensConstants.num_parts, 3)
        segment_angular_velocity = data_provider.xsens_data['segment_angularVelocity'].reshape(num_frames, XSensConstants.num_parts, 3)


    except KeyError as e:
        print(f"Error: Missing key in data_provider.xsens_data: {e}")
        print("Available keys:", data_provider.xsens_data.keys())
        return None
    except Exception as e:
        print(f"Error accessing or reshaping data from data_provider: {e}")
        return None

    # Create a mapping from Nymeria joint names to their index in the XSens data arrays
    xsens_name_to_index = {name: i for i, name in enumerate(XSensConstants.part_names)}
    # print("Available XSens/Nymeria Part Names and Indices:")
    # print(xsens_name_to_index)
    # print(f"Number of Nymeria parts: {XSensConstants.num_parts}")

    # Define target ProtoMotion skeleton
    proto_skeleton = get_proto_skeleton_tree()
    proto_node_names = proto_skeleton['node_names']
    proto_joint_count = len(proto_node_names)
    # print("\nTarget ProtoMotion Node Names and Indices:")
    # print({name: i for i, name in enumerate(proto_node_names)})
    # print(f"Number of ProtoMotion joints: {proto_joint_count}")


    # --- 2. Initialize ProtoMotion Data Arrays ---
    rotation_proto = np.zeros((num_frames, proto_joint_count, 4), dtype=np.float64) # Using WXYZ internally for now
    translation_proto = np.zeros((num_frames, proto_joint_count, 3), dtype=np.float64)
    # Initialize velocity arrays
    global_velocity = np.zeros((num_frames, proto_joint_count, 3), dtype=np.float64)
    global_angular_velocity = np.zeros((num_frames, proto_joint_count, 3), dtype=np.float64)


    # --- 4. Map Nymeria Data to ProtoMotion Structure ---
    print("\nMapping Nymeria joints to ProtoMotion joints:")
    mapping_summary = {} # To store what was mapped

    # Define the mapping based on your requirements and ProtoMotion joint names/indices
    # Format: { ProtoMotion_Joint_Name : (Source_Nymeria_Joint_Name, Source_Type) }
    # Source_Type can be 'direct', 'interpolated_trans', 'interpolated_rot'
    mapping_plan = {
        # Root and Spine
        'Pelvis': ('L5', 'direct'),           # Nymeria L5 -> Proto Pelvis (Index 0)
        'Torso': ('L3', 'direct'),            # Nymeria L3 -> Proto Torso (Index 9)
        'Spine': ('T12', 'direct'),           # Nymeria T12 -> Proto Spine (Index 10)
        'Chest': ('T8', 'direct'),            # Nymeria T8 -> Proto Chest (Index 11)
        # Head
        'Neck': ('Neck', 'direct'),           # Nymeria Neck -> Proto Neck (Index 12)
        'Head': ('Head', 'direct'),           # Nymeria Head -> Proto Head (Index 13)
        # Left Leg
        'L_Hip': ('L_UpperLeg', 'direct'),    # Nymeria L_UpperLeg -> Proto L_Hip (Index 1)
        'L_Knee': ('L_LowerLeg', 'direct'),   # Nymeria L_LowerLeg -> Proto L_Knee (Index 2)
        'L_Ankle': ('L_Foot', 'direct'),      # Nymeria L_Foot -> Proto L_Ankle (Index 3)
        'L_Toe': ('L_Toe', 'direct'),         # Nymeria L_Toe -> Proto L_Toe (Index 4)
        # Right Leg
        'R_Hip': ('R_UpperLeg', 'direct'),    # Nymeria R_UpperLeg -> Proto R_Hip (Index 5)
        'R_Knee': ('R_LowerLeg', 'direct'),   # Nymeria R_LowerLeg -> Proto R_Knee (Index 6)
        'R_Ankle': ('R_Foot', 'direct'),      # Nymeria R_Foot -> Proto R_Ankle (Index 7)
        'R_Toe': ('R_Toe', 'direct'),         # Nymeria R_Toe -> Proto R_Toe (Index 8)
        # Thorax/Collar (Using T8 as proxy like original code, adjust if needed)
        'L_Thorax': ('T8', 'direct'),         # Nymeria T8 -> Proto L_Thorax (Index 14)
        'R_Thorax': ('T8', 'direct'),         # Nymeria T8 -> Proto R_Thorax (Index 19)
        # Left Arm
        'L_Shoulder': ('L_Shoulder', 'direct'), # Nymeria L_Shoulder -> Proto L_Shoulder (Index 15)
        'L_Elbow': ('L_UpperArm', 'direct'),    # Nymeria L_UpperArm -> Proto L_Elbow (Index 16)
        'L_Wrist': ('L_Wrist', 'interpolated'), # Use interpolated data for Proto L_Wrist (Index 17)
        'L_Hand': ('L_Hand', 'direct'),         # Nymeria L_Hand -> Proto L_Hand (Index 18)
        # Right Arm
        'R_Shoulder': ('R_Shoulder', 'direct'), # Nymeria R_Shoulder -> Proto R_Shoulder (Index 20)
        'R_Elbow': ('R_UpperArm', 'direct'),    # Nymeria R_UpperArm -> Proto R_Elbow (Index 21)
        'R_Wrist': ('R_Wrist', 'interpolated'), # Use interpolated data for Proto R_Wrist (Index 22)
        'R_Hand': ('R_Hand', 'direct'),         # Nymeria R_Hand -> Proto R_Hand (Index 23)
    }

    # Apply the mapping plan
    for proto_idx, proto_name in enumerate(proto_node_names):
        if proto_name in mapping_plan:
            source_name, source_type = mapping_plan[proto_name]

            if source_type == 'direct':
                nymeria_idx = xsens_name_to_index.get(source_name)
                if nymeria_idx is not None and nymeria_idx < XSensConstants.num_parts:
                    try:
                        rotation_proto[:, proto_idx, :] = segment_quat_wxyz[:, nymeria_idx, :]
                        translation_proto[:, proto_idx, :] = segment_trans[:, nymeria_idx, :]
                        # Map velocity and angular velocity using the direct string keys
                        global_velocity[:, proto_idx, :] = segment_velocity[:, nymeria_idx, :]
                        global_angular_velocity[:, proto_idx, :] = segment_angular_velocity[:, nymeria_idx, :]
                        mapping_summary[proto_name] = f"Mapped from Nymeria '{source_name}' (Index {nymeria_idx}) with velocity and angular velocity"
                    except IndexError:
                         mapping_summary[proto_name] = f"Failed (IndexError) for Nymeria '{source_name}' (Index {nymeria_idx})"
                else:
                    mapping_summary[proto_name] = f"Failed (Nymeria source '{source_name}' not found or index invalid)"
            # Add logic for 'interpolated_trans', 'interpolated_rot' if needed for wrists
            # ... (Your existing or future interpolation logic for L_Wrist and R_Wrist)
            # For now, the wrists will show as "Failed (Not in mapping summary)" if only 'interpolated' is in the plan
        else:
            # This ProtoMotion joint wasn't in our plan (shouldn't happen if plan is complete)
             mapping_summary[proto_name] = "Failed (No mapping specified in plan)"


    # Print mapping summary
    for i, name in enumerate(proto_node_names):
        status = mapping_summary.get(name, "Failed (Not in mapping summary)")
        print(f"ProtoMotion Index {i} ({name}): {status}")

    # --- 5. Final Proto Data Structure ---
    # Extract root translation (mapped Pelvis, index 0)
    # Ensure it has shape (num_frames, 3)
    root_translation = translation_proto[:, 0, :]

    # Note: ProtoMotion often expects rotations as XYZW. Check format requirements.
    # If XYZW is needed, convert rotation_proto here:
    # rotation_proto_xyzw = rotation_proto[:, :, [1, 2, 3, 0]] # Convert WXYZ -> XYZW
    # Then use rotation_proto_xyzw in the 'rotation' field below.
    # For now, keeping WXYZ as derived from XSens. Adjust if your ProtoMotion consumer needs XYZW.

    proto_data = OrderedDict({
        # Using WXYZ format based on input. Change if ProtoMotion needs XYZW.
        'rotation': OrderedDict({'arr': rotation_proto, 'context': {'dtype': 'float64', 'format': 'wxyz'}}),
        'root_translation': OrderedDict({'arr': root_translation, 'context': {'dtype': 'float64'}}),
        'global_velocity': OrderedDict({'arr': global_velocity, 'context': {'dtype': 'float64'}}),
        'global_angular_velocity': OrderedDict({'arr': global_angular_velocity, 'context': {'dtype': 'float64'}}),
        'skeleton_tree': proto_skeleton, # Use the defined SMPL/Proto skeleton
        'is_local': False, # Assuming input data is global
        'fps': np.array([frame_rate], dtype=np.float64),
        '__name__': 'SkeletonMotion'
    })

    return proto_data

# The rest of the script (save_proto_npy and main) remains the same
def save_proto_npy(proto_data, output_file):
    """Saves the ProtoMotion data dictionary to a .npy file."""
    if proto_data:
        try:
            np.save(output_file, proto_data, allow_pickle=True) # Need allow_pickle for OrderedDict
            print(f"\nProtoMotion-formatted data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")
    else:
        print("No ProtoMotion data generated to save.")

def main():
    parser = argparse.ArgumentParser(description='Convert Nymeria data to ProtoMotion format with wrist interpolation and custom mapping.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing Nymeria .npy files (used by BodyDataProvider)')
    parser.add_argument('--output_file', type=str, default='proto_motion_mapped.npy', help='Path to save the ProtoMotion-formatted .npy file')
    parser.add_argument('--glb_file', type=str, default='', help='Optional GLB file path for BodyDataProvider (if needed by it)')

    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    # Create the data provider (assuming this function handles loading Nymeria data)
    data_provider = create_body_data_provider(args.data_dir, args.glb_file)

    if data_provider:
        print("Data provider created successfully.")
        # Generate ProtoMotion data using the new logic
        proto_motion_data = create_proto_motion_from_dataprovider(data_provider)
        if proto_motion_data:
            # Save the results
            save_proto_npy(proto_motion_data, args.output_file)
        else:
            print("Failed to generate ProtoMotion data.")
    else:
        print("Failed to create BodyDataProvider object. Check your data directory and dependencies.")

if __name__ == "__main__":
    main()