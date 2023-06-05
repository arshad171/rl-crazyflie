import time
import gym
import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from rl_crazyflie.envs.BalanceAviary import BalanceAviary

# define defaults
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "./results"

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 50
DEFAULT_DURATION_SEC = 2
DEFAULT_CONTROL_FREQ_HZ = 48

INIT_XYZS = np.array([[0.0, 0.0, 1.0] for _ in range(DEFAULT_NUM_DRONES)])
INIT_RPYS = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
NUM_PHYSICS_STEPS = 1


def main():
    balance_env = gym.make(
        "balance-aviary-v0",
        **{
            "drone_model": DEFAULT_DRONES,
            "initial_xyzs": INIT_XYZS,
            "initial_rpys": INIT_RPYS,
            "freq": DEFAULT_SIMULATION_FREQ_HZ,
            "aggregate_phy_steps": NUM_PHYSICS_STEPS,
            "gui": DEFAULT_GUI,
            "record": DEFAULT_RECORD_VIDEO,
            "ext_dist_mag": None,
            "flip_freq": None,
        },
    )
    balance_env.reset()

    while True:
        balance_env.step(np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))


if __name__ == "__main__":
    main()
