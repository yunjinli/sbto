"""Visualize DFQ hand collision geoms statically (no physics).
Press F in viewer to toggle collision geom wireframes.
Close the window to exit.
"""

import os
import mujoco
import mujoco.viewer

import sys

SCENE_DIR = "sbto/models/unitree_g1"
TMP_XML = os.path.join(SCENE_DIR, "_debug_scene.xml")

XML = """\
<mujoco>
  <include file="scene_mjx_29dof_with_dfq_hands.xml"/>
  <worldbody>
    <body name="empty_body" pos="0.4 0 0.1">
      <freejoint/>
      <geom name="obj" type="box" size="0.155 0.155 0.17" pos="0 0 0.0105"
            rgba="0.3 0.3 0.3 1" mass="0.6"
            condim="3" contype="0" conaffinity="1"
            solref="0.04 1." friction="0.6 0.0001 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""

with open(TMP_XML, "w") as f:
    f.write(XML)

try:
    model = mujoco.MjModel.from_xml_path(TMP_XML)
finally:
    os.remove(TMP_XML)

data = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

# Forward kinematics only — no physics, robot stays in place
mujoco.mj_kinematics(model, data)

print("Press F to toggle collision geom wireframes.")
print("Close the window to exit.")

# launch() blocks until window is closed; no manual loop needed
mujoco.viewer.launch(model, data)
