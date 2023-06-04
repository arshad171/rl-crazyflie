# RL CrazyFlie

## Requirements

1. Install `gym-pybullet-drones` in editable mode.
2. No need to explicitely install `stable-baselines3`, it should be pulled as a dep by `gym-pybullet-drones`.

## URDF

Run `xacro stick.xacro > stick.urdf` to generate the URDF file. Copy the URDF to `gym-pybullet-drones/gym_pybullet_drones/assets/` so gym can access it.

## pybullet

- `id = p.loadURDF("path")` returns the id of the object (needed for interaction).
- `pos, quat = p.getBasePositionAndOrientation(id, physicsClientId=client)` returns the position and orientation of the base/root link. The `id` param is the id returned when loading the URDF.
- `rpy = p.getEulerFromQuaternion(quat)` converts quat to rpy.