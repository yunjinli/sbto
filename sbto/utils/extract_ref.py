import pickle
import mujoco
import numpy as np
from scipy.interpolate import interp1d

def compute_time_from_fps(fps, N):
    time = np.arange(N) / fps
    return time

def quatxyzw2quatwxyz(quat):
    new_quat = np.empty_like(quat)
    new_quat[:, 0] = quat[:, -1]
    new_quat[:, 1:] = quat[:, :3]
    return new_quat

def normalize_quat(quat):
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat

def concatenate_full_state(data):
    if "object_rot" in data and "object_root_pos" in data:
        qpos = np.concatenate(
            (
                data["root_pos"],
                data["root_rot"],
                data["dof_pos"],
                data["object_root_pos"],
                data["object_rot"],
            ), axis=-1
        )
        qvel = np.zeros((qpos.shape[0], qpos.shape[-1] - 2))

    else:
        qpos = np.concatenate(
            (
                data["root_pos"],
                data["root_rot"],
                data["dof_pos"],
            ), axis=-1
        )
        qvel = np.zeros((qpos.shape[0], qpos.shape[-1] - 1))

    x = np.concatenate((qpos, qvel), axis=-1)
    return qpos, qvel, x

def interpolate_data(data, dt: float):
    t_interp = np.arange(0, data["time"][-1], dt)
    NO_INTERP = ["time", "fps"]
    for k, v in data.items():
        if k not in NO_INTERP and v is not None:
            print(k)
            interpolate = interp1d(
                data["time"],
                y = v,
                axis=0
            )
            data[k] = interpolate(t_interp)
            if "rot" in k:
                data[k] = normalize_quat(data[k])

    data["time"] = t_interp
    return data

def extract_sensor_data(mj_model, state_traj, sensor_names):
    """
    Extract specified sensor data from a MuJoCo model along a state trajectory.

    Args:
        mj_model_xml_path (str): Path to the MuJoCo XML model.
        state_traj (np.ndarray): Array of shape [T, nq + nv] containing qpos and qvel
                                 for each time step.
        sensor_names (list[str]): List of sensor names to extract.

    Returns:
        np.ndarray: Array of shape [T, len(sensor_names), sensor_dim_i]
                    containing the sensor data per timestep.
                    If sensor dims differ, returns a list of arrays instead.
    """
    # Load model and create data
    data = mujoco.MjData(mj_model)

    # Get sensor indices and dimensions
    sensor_info = []
    for name in sensor_names:
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = mj_model.sensor_adr[sid]
        dim = mj_model.sensor_dim[sid]
        sensor_info.append((adr, dim))

    T = len(state_traj)
    sensor_data = []

    for t in range(T):
        # Split qpos and qvel
        nq = mj_model.nq
        data.qpos[:] = state_traj[t, :nq]
        data.qvel[:] = state_traj[t, nq:]
        
        # Compute sensors
        mujoco.mj_forward(mj_model, data)
        
        # Extract requested sensors
        step_sensors = []
        for adr, dim in sensor_info:
            step_sensors.append(np.copy(data.sensordata[adr:adr+dim]))
        sensor_data.append(step_sensors)

    # Convert to array if possible
    try:
        sensor_data = np.squeeze(np.array(sensor_data))
    except:
        # fall back to list of per-step arrays if dims differ
        pass

    return sensor_data


def load_reference_trajectory(
    path: str,
    xml_path: str,
    sensor_names = [],
    speedup: float = 1.,
    z_offset: float = 0.
    ):
    # if the file contains multiple pickled objects in sequence:
    objs = []
    with open(path, "rb") as f:
        while True:
            try:
                objs.append(pickle.load(f))
            except EOFError:
                break

    data = {}
    for sub_dict in objs:
        for k, v in sub_dict.items():
            data[k] = v

    data["time"] = compute_time_from_fps(data["fps"] * speedup, len(data["root_pos"]))

    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    dt_interp = mj_model.opt.timestep
    if dt_interp > 0.:
        data = interpolate_data(data, dt_interp)

    if z_offset != 0:
        data["root_pos"][:, 2] = data["root_pos"][:, 2] - z_offset
        if "object_pos" in data:
            data["object_root_pos"][:, 2] = data["object_root_pos"][:, 2] - z_offset

    data["root_rot"] = quatxyzw2quatwxyz(data["root_rot"])
    if "object_rot" in data:
        data["object_rot"] = quatxyzw2quatwxyz(data["object_rot"])
        
    data["qpos"], data["qvel"], data["x"] = concatenate_full_state(data)

    for sensor_name in sensor_names:
        data[sensor_name] = extract_sensor_data(mj_model, data["x"], [sensor_name])

    return data

if __name__ == "__main__":
    import mujoco
    from sbto.utils.viewer import render_and_save_trajectory, visualize_trajectory

    path = "test/sub3_largebox_003.pkl"
    xml = "sbto/models/unitree_g1/scene_29dof.xml"

    data = load_reference_trajectory(path, xml, speedup=1.8, z_offset=0.023)
    print(data.keys())
    print(data['x'].shape)

    mj_model = mujoco.MjModel.from_xml_path(xml)
    mj_data = mujoco.MjData(mj_model)
    
    visualize_trajectory(mj_model, mj_data, data["time"], data["x"])
    # render_and_save_trajectory(mj_model, mj_data, data["time"], data["x"], save_path="test/test.mp4", fps=30)


