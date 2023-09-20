# RL CrazyFlie

## Repository

```bash
.
├── README.md
├── results_models                          # explicitely saved models/results
│   └── navigation
├── results_plots                           # analysis results
│   └── navigation
├── rl-crazyflie.code-workspace
├── rl_crazyflie                            # module
│   ├── __init__.py
│   ├── __pycache__
│   ├── assets                              # assets: URDFs
│   ├── envs                                # custom envs
│   └── utils
├── scripts                                 # scripts to analyse/plot results
│   ├── calc_intertia.py
│   ├── err_mag.ipynb
│   ├── tb_logs.ipynb
│   └── viz_log_traj.ipynb
├── test_bal.py                             # train/test balance-aviary
├── test_mo_bal.py                          # train/test multi-objective balance-aviary
├── test_nav.py                             # train/test navigation-aviary
└── test_nav_err.py                         # train/test navigation-aviary with action feedback
```

## Requirements

### Single-objective RL
1. Install `gym-pybullet-drones` fork from here [arshad171/gym-pybullet-drones:rl-enhancements](https://github.com/arshad171/gym-pybullet-drones). The `rl-enhancements` branch has some nifty enhancements for experiments and multi-objective RL.
It is ideal to install the package in "editable" mode.
The installation is broken with latest updates of `pip`, `setuptools` and `wheel`. Install these specific versions (recommended by gym-pybullet-drones):
```
pip install --upgrade pip==23.0.1
pip install wheel==0.38.4 --upgrade
pip install setuptools==66 --upgrade
```
2. No need to explicitely install `stable-baselines3`, it should be pulled as a dep by `gym-pybullet-drones`.

3. Install `sb3-contrib, shimmy` for the LSTM version of PPO, `pip install sb3-contrib shimmy`. Ignore `stable-baselines3` version conflict.

### Multi-objective RL
1. Install `gym-pybullet-drones` (above).
1. `pip install "mo-gymnasium[all]"` to install multi-objective gymnasium. [Farama-Foundation/MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium.git).
1. Install the multi-objective version of stable_baselines from here: [LucasAlegre/morl-baselines](https://github.com/LucasAlegre/morl-baselines.git).
1. Install `pylibcdd` for MORL-baselines from here [cdd](https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation). The package requires dev tools, check the link for more infomation.
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

- [x] Train longer for better stabilization.

- [x] Errors vs Time

- [ ] Init agent at destination and compare the deviations when subject to wind.

- [x] Smoothness $\int |\tau''(t)|^2 dt$, sum of second derivaties.

- [x] Steady-state errors.

- [x] Table highlights.