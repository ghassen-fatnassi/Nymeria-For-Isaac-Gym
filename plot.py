# Assuming you have the BodyDataProvider class defined in a file like 'body_data_provider.py'
from nymeria_files.body_motion_provider import create_body_data_provider
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from nymeria_files.xsens_constants import XSensConstants
# Initialize the BodyDataProvider
data_provider = create_body_data_provider("xdata.npz", "xdata_blueman.glb")
if data_provider:
    timestamps_us = data_provider.xsens_data[XSensConstants.k_timestamps_us]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')

    for t_us in timestamps_us[::10]: # Plot every 10th frame for speed
        skeleton_data, _ = data_provider.get_posed_skeleton_and_skin(int(t_us))

        if skeleton_data is not None:
            ax.clear()
            for bone in skeleton_data:
                ax.plot([bone[0, 0], bone[1, 0]],
                        [bone[0, 1], bone[1, 1]],
                        [bone[0, 2], bone[1, 2]], c='b')

            # Set reasonable limits based on the first frame
            if t_us == timestamps_us[0]:
                all_points = skeleton_data.reshape(-1, 3)
                min_val = np.min(all_points)
                max_val = np.max(all_points)
                ax.set_xlim(min_val - 0.1, max_val + 0.1)
                ax.set_ylim(min_val - 0.1, max_val + 0.1)
                ax.set_zlim(min_val - 0.1, max_val + 0.1)

            plt.pause(0.01)

    plt.show()
else:
    print("Failed to load body data.")