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

## Notes

- physical parameters of stick (mass, length, rad): too much => sinks the drone, too light => gets tossed away

- open-loop vs closed-loop: closed-loop introduces too much of control delay => too few RL steps

- rewards:
    - terminate the episode immediately when the stick falls => else noisy reward. 
    - a small penatly for moving drifting away from the initial state (along z) => otherwise the agent would learn to simply brace the fall and balance the stick.

- initializing stick at an angle helps after the agent is trained for 1e6

- algorithmic convergence: stochastic policies (PPO, A2C) were found to converge better than deterministic (TD3).

## Observations

- Model with integral errors has a smoother path, while the other models have a zig-zag path with the agents taking long strides. But the error model does not stabilize well after reaching the destination.

- The magnitude of the error decreases after a few RL steps.

- The model trained on dist (without error) tends to overfit the training scenario (+ve dist), learns maneuvers with max velocity. This does not work well when the direction of wind is reversed or agent is initialized to different point.

## Future

- Train longer for better stabilization.

- Errors vs Time

- Init agent at destination and compare the deviations when subject to wind.

- Smoothness $\int |\tau''(t)^2| dt$, sum of second derivaties.

- Steady-state errors.

- Table highlights.