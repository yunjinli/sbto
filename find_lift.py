import glob
import numpy as np

# Path to your dataset
files = glob.glob("datasets/robot-object/*.npz")

print("Scanning for lifting sequences...\n")
for f in files:
    data = np.load(f, mmap_mode="r")
    qpos = data["qpos"]

    # In OmniRetarget, the object pose is the last 7 dimensions: [qw, qx, qy, qz, x, y, z]
    # So the Z-height is the very last element (index -1)
    z_heights = qpos[:, -1]

    # Check if the object is lifted more than 5cm from its starting height
    max_lift = np.max(z_heights) - z_heights[0]

    if max_lift > 0.05:
        print(f"LIFT DETECTED: {f} (Lifted by {max_lift * 100:.1f} cm)")
