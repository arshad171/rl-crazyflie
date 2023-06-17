import sys

sys.path.append("./src")
sys.path.append("./src/rl")

import os
import pickle
import time

import pandas as pd
import gym
import numpy as np
import torch
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

DIR = "results-nav"

MODEL_PATH = f"./{DIR}/model"
ENV_PATH = f"./{DIR}/env"
LOGS_PATH = f"./{DIR}/logs"
TB_LOGS_PATH = f"./{DIR}/logs"
PLT_LOGS_PATH = f"./{DIR}/plt"

# define defaults
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = f"./{DIR}/rec"

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 50
DEFAULT_DURATION_SEC = 2
DEFAULT_CONTROL_FREQ_HZ = 48

INIT_XYZS = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
INIT_RPYS = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
NUM_PHYSICS_STEPS = 1

PERIOD = 10

# "train" / "test"
MODE = Modes.TRAIN_TEST

NUM_EVAL_EPISODES = 1
TEST_EXT_DIST_X_MAX = 0.1
TEST_EXT_DIST_XYZ_MAX = 0.05
TEST_EXT_DIST_STEPS = 10

FLIP_FREQ = -1 if MODE == Modes.TRAIN else 20

# hyperparams for training
NUM_EPISODES = 1e6
ACTOR_NET_ARCH = [50, 100, 500, 100, 50]
CRITIC_NET_ARCH = [50, 100, 500, 100, 50]
TRAIN_EXT_DIST = np.array([
    [0.0, 0.0, 0.0],
    [0.05, 0.0, 0.0],
    [0.0, 0.0, 0.05],
    [0.025, 0.025, 0.025],
])




def run(dist):
    nav_env = gym.make(
        "navigation-aviary-err-v0",
        **{
            "drone_model": DEFAULT_DRONES,
            "initial_xyzs": INIT_XYZS,
            "initial_rpys": INIT_RPYS,
            "freq": DEFAULT_SIMULATION_FREQ_HZ,
            "aggregate_phy_steps": NUM_PHYSICS_STEPS,
            "gui": DEFAULT_GUI,
            "record": DEFAULT_RECORD_VIDEO,
            "ext_dist_mag": dist,
            "flip_freq": FLIP_FREQ,
        },
    )

    # nav_env = Monitor(nav_env, TB_LOGS_PATH)

    if MODE == Modes.TRAIN or MODE == Modes.TRAIN_TEST:
        n_actions = nav_env.action_space.shape[-1]
        mu = np.zeros(n_actions)
        sigma = 0.5 * np.ones(n_actions)

        new_logger = configure(LOGS_PATH, ["stdout", "csv", "tensorboard"])
        model = PPO(
            "MlpPolicy",
            nav_env,
            # policy_kwargs=dict(net_arch=dict(pi=ACTOR_NET_ARCH, qf=CRITIC_NET_ARCH)),
            verbose=0,
            # action_noise=NormalActionNoise(mu, sigma),
            tensorboard_log=TB_LOGS_PATH,
        )

        # # resume training
        # nav_env = pickle.load(open(ENV_PATH, "rb"))
        # model = PPO.load(MODEL_PATH, nav_env)

        model.set_logger(new_logger)
        model.learn(
            total_timesteps=NUM_EPISODES,
            # log_interval=1,
            # callback=TBCallback(log_dir=TB_LOGS_PATH),
        )

        # save model
        model.save(MODEL_PATH)
        pickle.dump(nav_env, open(ENV_PATH, "wb"))

        return None

    elif MODE == Modes.TEST or MODE == Modes.TRAIN_TEST:
        # nav_env = pickle.load(open(ENV_PATH, "rb"))
        model = PPO.load(MODEL_PATH, nav_env)
        # nav_env = model.get_env()

        logger = Logger(
            logging_freq_hz=int(nav_env.SIM_FREQ / nav_env.AGGR_PHY_STEPS),
            num_drones=1,
            output_folder=PLT_LOGS_PATH,
        )

        # simulation
        # rewards = evaluate_policy(model, nav_env, n_eval_episodes=3, return_episode_rewards=True)
        mean_eps_reward, std_eps_reward = evaluate_policy(
            model, nav_env, n_eval_episodes=NUM_EVAL_EPISODES, render=False
        )
        mean_step_reward = mean_eps_reward / (DEFAULT_DURATION_SEC * nav_env.SIM_FREQ)

        print(f"{mean_eps_reward=} | {std_eps_reward=} | {mean_step_reward=}")

        optimal_controller = DSLPIDControl(drone_model=DEFAULT_DRONES)
        ctrl = [optimal_controller for _ in range(DEFAULT_NUM_DRONES)]

        next_obs = nav_env.reset()

        # action, _ = model.predict(next_obs)
        # print(model.actor(next_obs)[0])

        distance_travelled = 0.0
        prev_state = np.zeros(shape=(3,))
        START = time.time()
        for i in range(
            0, int(DEFAULT_DURATION_SEC * nav_env.SIM_FREQ), NUM_PHYSICS_STEPS
        ):
            action, _ = model.predict(next_obs)
            next_obs, reward, done, info = nav_env.step(action)
            distance_travelled += np.linalg.norm(next_obs[:3] - prev_state)
            prev_state = next_obs[:3]
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
                    np.linspace(0.0, TEST_EXT_DIST_X_MAX, TEST_EXT_DIST_STEPS),
                    np.zeros(shape=(TEST_EXT_DIST_STEPS,)),
                    np.zeros(shape=(TEST_EXT_DIST_STEPS,)),
                ]
            ).transpose(),
            "z": np.vstack(
                [
                    np.zeros(shape=(TEST_EXT_DIST_STEPS,)),
                    np.zeros(shape=(TEST_EXT_DIST_STEPS,)),
                    np.linspace(0.0, TEST_EXT_DIST_X_MAX, TEST_EXT_DIST_STEPS),
                ]
            ).transpose(),
            "xyz": np.vstack(
                [
                    np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS),
                    np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS),
                    np.linspace(0.0, TEST_EXT_DIST_XYZ_MAX, TEST_EXT_DIST_STEPS),
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
        for dir in ext_dists:
            for i in range(TEST_EXT_DIST_STEPS):
                dist = ext_dists[dir][i, :]

                mean_eps_reward, std_eps_reward, mean_step_reward, distance_travelled = run(dist=dist)

                # ext_dists_res_df = pd.concat([
                #     ext_dists_res_df,
                #     pd.DataFrame(
                #         {
                #             "dir": dir,
                #             "dist": str(dist),
                #             "mean_eps_reward": mean_eps_reward,
                #             "std_eps_reward": std_eps_reward,
                #             "mean_step_reward": mean_step_reward,
                #         },
                #     ),
                # ])

                ext_dists_res_df.append(
                    {
                        "dir": dir,
                        "dist": dist,
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
