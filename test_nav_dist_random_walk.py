###
# dists, no error
###

import copy
import sys

sys.path.append("./src")
sys.path.append("./src/rl")

import os
import pickle
import time

import pandas as pd
import gym
import numpy as np
import torch as th
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

# from gym_pybullet_drones.utils.Logger import Logger
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from gym.envs.registration import register

from rl_crazyflie.envs.NavigationAviaryErr import NavigationAviaryErr
from rl_crazyflie.utils.Logger import Logger
from rl_crazyflie.utils.constants import Modes

# from plotter import plot

DIR = "nav-results-random_walk"

MODEL_PATH = f"./{DIR}/model"
ENV_PATH = f"./{DIR}/env"
LOGS_PATH = f"./{DIR}/logs"
TB_LOGS_PATH = f"./{DIR}/logs"
PLT_LOGS_PATH = f"./{DIR}/plt/it"

# define defaults
VIEW = True
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = f"./{DIR}/rec"

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 100
DEFAULT_DURATION_SEC = 5
DEFAULT_CONTROL_FREQ_HZ = 48

INIT_XYZS_TRAIN = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
INIT_XYZS_TEST = np.array([[0.0, 0.0, 1.0] for _ in range(DEFAULT_NUM_DRONES)])
INIT_RPYS = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
NUM_PHYSICS_STEPS = 1

PERIOD = 10

# "train" / "test"
MODE = Modes.TEST

NUM_EVAL_EPISODES = 1
TEST_EXT_DIST_X_MAX = 0.1
TEST_EXT_DIST_XYZ_MAX = 0.05
TEST_EXT_DIST_STEPS = 3

FLIP_FREQ = 20

# hyperparams for training
NUM_EPISODES = 1e6
ACTOR_NET_ARCH = [50, 100, 500, 100, 50]
CRITIC_NET_ARCH = [50, 100, 500, 100, 50]
TRAIN_EXT_DIST = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [0.0, 0.0, 0.05],
        [0.0, 0.0, -0.05],
        [0.025, 0.025, 0.025],
        [-0.025, -0.025, -0.025],
    ]
)


def run(dist, dir=None):
    if MODE == Modes.TEST:
        global FLIP_FREQ

        os.makedirs(PLT_LOGS_PATH, exist_ok=True)

        nav_env = gym.make(
            "navigation-aviary-v0",
            **{
                "drone_model": DEFAULT_DRONES,
                "initial_xyzs": INIT_XYZS_TEST,
                "initial_rpys": INIT_RPYS,
                "freq": DEFAULT_SIMULATION_FREQ_HZ,
                "aggregate_phy_steps": NUM_PHYSICS_STEPS,
                "gui": False,
                "record": False,
                "ext_dist_mag": dist,
                "flip_freq": FLIP_FREQ,
                "output_folder": DEFAULT_OUTPUT_FOLDER,
            },
        )

        # nav_env = pickle.load(open(ENV_PATH, "rb"))
        # model = PPO.load(MODEL_PATH, nav_env)
        model = PPO(
            "MlpPolicy",
            nav_env,
            # policy_kwargs=dict(net_arch=dict(pi=ACTOR_NET_ARCH, qf=CRITIC_NET_ARCH)),
            verbose=0,
            # action_noise=NormalActionNoise(mu, sigma),
            tensorboard_log=TB_LOGS_PATH,
        )
        # nav_env = model.get_env()

        # logger = Logger(
        #     logging_freq_hz=int(nav_env.SIM_FREQ / nav_env.AGGR_PHY_STEPS),
        #     num_drones=1,
        #     output_folder=PLT_LOGS_PATH,
        # )

        if not VIEW:
            # simulation
            # rewards = evaluate_policy(model, nav_env, n_eval_episodes=3, return_episode_rewards=True)
            mean_eps_reward, std_eps_reward = evaluate_policy(
                model, nav_env, n_eval_episodes=NUM_EVAL_EPISODES, render=False
            )
            mean_step_reward = mean_eps_reward / (DEFAULT_DURATION_SEC * nav_env.SIM_FREQ)

            print(f"{mean_eps_reward=} | {std_eps_reward=} | {mean_step_reward=}")
        else:
            mean_eps_reward = 0
            std_eps_reward = 0
            mean_step_reward = 0

        optimal_controller = DSLPIDControl(drone_model=DEFAULT_DRONES)
        ctrl = [optimal_controller for _ in range(DEFAULT_NUM_DRONES)]

        next_obs = nav_env.reset()

        # action, _ = model.predict(next_obs)
        # print(model.actor(next_obs)[0])

        coordinates = []

        distance_travelled = 0.0
        prev_state = np.zeros(shape=(3,))
        START = time.time()
        for i in range(
            0, int(DEFAULT_DURATION_SEC * nav_env.SIM_FREQ), NUM_PHYSICS_STEPS
        ):
            log = {
                "x": next_obs[0],
                "y": next_obs[1],
                "z": next_obs[2],
                "roll": next_obs[3],
                "pitch": next_obs[4],
                "yaw": next_obs[5],
                "action_mag": None,
                "xe": None,
                "ye": None,
                "ze": None,
            }

            # temp_old_state = next_obs

            prev_obs = next_obs[:3]

            # action, _ = model.predict(next_obs)
            # original: [-0.05, 0.05]
            # pred: [-1, 1]
            action = np.random.uniform(low=-1.0, high=1.0, size=(4,))
            action[3] = -1
            print(action)


            next_obs, reward, done, info = nav_env.step(action)
            nav_env.ctrl.reset()
            distance_travelled += np.linalg.norm(next_obs[:3] - prev_state)
            prev_state = next_obs[:3]

            action_temp = action
            action_temp *= 0.05

            log["action_mag"] = np.linalg.norm(action_temp)

            log["xe"] = next_obs[0] - (prev_obs[0] + action_temp[0])
            log["ye"] = next_obs[1] - (prev_obs[1] + action_temp[1])
            log["ze"] = next_obs[2] - (prev_obs[2] + action_temp[2])

            coordinates.append(log)

            # print("*"*10)

            # temp_state = copy.deepcopy(next_obs)
            # print("state: ", temp_state)
            # temp_action_ze, _ = model.predict(temp_state, deterministic=True)
            # temp_state[12:] = 0
            # temp_action_e, _ = model.predict(temp_state, deterministic=True)

            # print("action_ze: ", 0.05  * temp_action_ze)
            # print("action_e: ", 0.05 * temp_action_e)

            # print("old state: ", temp_old_state[:3])
            # print("next state: ", temp_old_state[:3] + 0.05 * temp_action_e)
            # print("curr state: ", next_obs[:3])
            # print("error (th): ", next_obs[:3] - (temp_old_state[:3] + 0.05 * temp_action_e))
            # print("error (state): ", next_obs[12:])

            # print("*"*10)

            # print(action)


            # logger.log(
            #     drone=0,
            #     timestamp=i / nav_env.SIM_FREQ,
            #     state=np.hstack(
            #         [
            #             next_obs[0:3],
            #             next_obs[10:13],
            #             next_obs[7:10],
            #             np.resize(action, (4)),
            #             [0, 0, 0, 0, 0, 0, 0, 0],
            #         ]
            #     ),
            # )

            # if done:
            #     nav_env.reset()

            if i % nav_env.SIM_FREQ == 0:
                nav_env.render()

            if DEFAULT_GUI:
                sync(i, START, nav_env.TIMESTEP)

        nav_env.reset()
        nav_env.close()
        del nav_env

        df_coordinates = pd.DataFrame(coordinates)

        df_coordinates.to_csv(os.path.join(PLT_LOGS_PATH, f"xyz_all.csv"), index=False)

        # logger.save_as_csv(comment="test")
        # plot()

        return mean_eps_reward, std_eps_reward, mean_step_reward, distance_travelled


if __name__ == "__main__":
    if MODE == Modes.TRAIN or MODE == Modes.TRAIN_TEST:
        run(TRAIN_EXT_DIST)
    elif MODE == Modes.TEST or MODE == Modes.TRAIN_TEST:
        ext_dists = {
            "x": np.vstack(
                [
                    np.hstack([np.linspace(0.0, TEST_EXT_DIST_X_MAX, TEST_EXT_DIST_STEPS), -1 * np.linspace(0.0, TEST_EXT_DIST_X_MAX, TEST_EXT_DIST_STEPS)]),
                    np.hstack([np.zeros(shape=(TEST_EXT_DIST_STEPS,)), np.zeros(shape=(TEST_EXT_DIST_STEPS,))]),
                    np.hstack([np.zeros(shape=(TEST_EXT_DIST_STEPS,)), np.zeros(shape=(TEST_EXT_DIST_STEPS,))]),
                ]
            ).transpose(),
            "z": np.vstack(
                [
                    np.hstack([np.zeros(shape=(TEST_EXT_DIST_STEPS,)), np.zeros(shape=(TEST_EXT_DIST_STEPS,))]),
                    np.hstack([np.zeros(shape=(TEST_EXT_DIST_STEPS,)), np.zeros(shape=(TEST_EXT_DIST_STEPS,))]),
                    np.hstack([np.linspace(0.0, TEST_EXT_DIST_X_MAX, TEST_EXT_DIST_STEPS), -1 * np.linspace(0.0, TEST_EXT_DIST_X_MAX, TEST_EXT_DIST_STEPS)]),
                ]
            ).transpose(),
            "xyz": np.vstack(
                [
                    np.hstack([np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS), -1 * np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS)]),
                    np.hstack([np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS), -1 * np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS)]),
                    np.hstack([np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS), -1 * np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS)]),
                ]
            ).transpose(),
        }

        ext_dists_res_df = []
        # ext_dists_res_df = pd.DataFrame(
        #     columns=[
        #         "dir",
        #         "dist",
        #         "mean_eps_reward",
        #         "std_eps_reward",
        #         "mean_step_reward",
        #     ]
        # )

        (
            mean_eps_reward,
            std_eps_reward,
            mean_step_reward,
            distance_travelled,
        ) = run(dist=TRAIN_EXT_DIST, dir=None)

        ext_dists_res_df.append(
            {
                "dir": "all",
                "dist": "all",
                "mean_eps_reward": mean_eps_reward,
                "std_eps_reward": std_eps_reward,
                "mean_step_reward": mean_step_reward,
                "distance_travelled": distance_travelled,
            },
            # ignore_index=True,
        )

        ext_dists_res_df = pd.DataFrame(ext_dists_res_df)
        ext_dists_res_df.to_csv(
            path_or_buf=os.path.join(PLT_LOGS_PATH, "ext_dists_res_df.csv")
        )
