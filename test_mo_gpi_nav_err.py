import os
import json
import time

import dill
import numpy as np
import torch as th

import pandas as pd
import gym
import mo_gymnasium as mo_gym

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

from morl_baselines.common.evaluation import eval_mo, eval_mo_reward_conditioned
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPILSContinuousAction,
    GPIPDContinuousAction,
)
from morl_baselines.single_policy.ser.mo_ppo import make_env
from morl_baselines.common.utils import make_gif

from rl_crazyflie.envs.MONavigationAviaryErr import MONavigationAviaryErr
from rl_crazyflie.utils.constants import Modes
from rl_crazyflie.utils.numpy_encoder import NumpyEncoder



DIR = "nav-results-mo-err"

MODEL_PATH = f"./{DIR}/model"
ENV_PATH = f"./{DIR}/env"
LOGS_PATH = f"./{DIR}/logs"
TB_LOGS_PATH = f"./{DIR}/logs"
PLT_LOGS_PATH = f"./{DIR}/plt/it"

# define defaults
VIEW = False
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = f"./{DIR}"

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 50
DEFAULT_DURATION_SEC = 2
DEFAULT_CONTROL_FREQ_HZ = 48

INIT_XYZS_TRAIN = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
INIT_XYZS_TEST = np.array([[2.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
INIT_RPYS = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
NUM_PHYSICS_STEPS = 1

PERIOD = 10

# "train" / "test"
MODE = Modes.TRAIN_TEST

NUM_EVAL_EPISODES = 1
TEST_EXT_DIST_X_MAX = 0.1
TEST_EXT_DIST_XYZ_MAX = 0.05
TEST_EXT_DIST_STEPS = 3

FLIP_FREQ = 20

# hyperparams for training
NUM_EPISODES = 1e6
NUM_ENVS = 4 # 4
POP_SIZE = 6 # 6
WARMUP_ITERATIONS = 40 # 80
EVOLUTIONARY_ITERATIONS = 10 # 20
NET_ARCH = [64, 64] # [256, 256]
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

WEIGHT_SUPPORTS = [
        np.array([1.0, 0.0]),
        # np.array([1.0, 0.5]),
        np.array([1.0, 0.25]),
    ]

if __name__ == "__main__":
    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)

    env_id = "mo-navigation-aviary-err-v0"

    if MODE == Modes.TRAIN or MODE == Modes.TRAIN_TEST:
        ref_point = np.array([-100.0, -100.0])

        eval_env = mo_gym.make(
        env_id,
            **{
                "drone_model": DEFAULT_DRONES,
                "initial_xyzs": INIT_XYZS_TRAIN,
                "initial_rpys": INIT_RPYS,
                "freq": DEFAULT_SIMULATION_FREQ_HZ,
                "aggregate_phy_steps": NUM_PHYSICS_STEPS,
                "record": DEFAULT_RECORD_VIDEO,
                "ext_dist_mag": TRAIN_EXT_DIST,
                "flip_freq": FLIP_FREQ,
                "gui": False,
                "output_folder": DEFAULT_OUTPUT_FOLDER,
            },
        )

        algo = GPILSContinuousAction(
            env=eval_env,
            # origin=ref_point,
            # gamma=0.99,
            project_name="mo-nav-err",
            log=True,
            seed=0,
            buffer_size=1e6,
            # num_envs=NUM_ENVS,
            # pop_size=POP_SIZE,
            # warmup_iterations=WARMUP_ITERATIONS,
            # evolutionary_iterations=EVOLUTIONARY_ITERATIONS,
            # net_arch=NET_ARCH
        )

        algo.set_weight_support(WEIGHT_SUPPORTS)

        pf = algo.train(
            total_timesteps=int(NUM_EPISODES),
            eval_env=eval_env,
            ref_point=ref_point,
            known_pareto_front=None,
        )

        dill.dump(eval_env, open(ENV_PATH, "wb"))
        dill.dump(algo, open(MODEL_PATH, "wb"))

    if MODE == Modes.TEST or MODE == Modes.TRAIN_TEST:
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

        for dir in ext_dists:
            for i in range(2 * TEST_EXT_DIST_STEPS):
                # load_env = dill.load(open(ENV_PATH, "rb"))
                load_algo = dill.load(open(MODEL_PATH, "rb"))

                dist = ext_dists[dir][i, :]

                metrics = []
                for ix, agent_weights in enumerate(WEIGHT_SUPPORTS):
                    os.makedirs(os.path.join(PLT_LOGS_PATH, f"agent_{ix}"), exist_ok=True)

                    # eval_env = dill.load(open(ENV_PATH, "rb"))

                    eval_env = mo_gym.make(
                        env_id,
                        **{
                            "drone_model": DEFAULT_DRONES,
                            "initial_xyzs": INIT_XYZS_TEST,
                            "initial_rpys": INIT_RPYS,
                            "freq": DEFAULT_SIMULATION_FREQ_HZ,
                            "aggregate_phy_steps": NUM_PHYSICS_STEPS,
                            "record": False,
                            "ext_dist_mag": dist,
                            "flip_freq": -1,
                            "eval_reward": True,
                            "gui": False,
                            "output_folder": DEFAULT_OUTPUT_FOLDER,
                        },
                    )

                    eval_env.reset()
                    # w -> weight vector for discounted reward
                    scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
                        agent=load_algo, env=eval_env, w=agent_weights, render=False
                    )
                    metrics.append({
                        "id": ix,
                        "dir": dir,
                        "dist": dist,
                        "weights": agent_weights.tolist(),
                        "scalarized_rew": float(scalarized),
                        "discounted_scalarized_rew":float(discounted_scalarized),
                        "vector_rew": reward.tolist(),
                        "discounted_vector_rew": discounted_reward.tolist(),
                    })
                    print(f"Agent #{ix}")
                    print(f"Agent weights: {agent_weights}")
                    print(f"Scalarized: {scalarized}")
                    print(f"Discounted scalarized: {discounted_scalarized}")
                    print(f"Vectorial: {reward}")
                    print(f"Discounted vectorial: {discounted_reward}")
                    print("-----")

                    next_obs, _ = eval_env.reset()
                    coordinates = []
                    distance_travelled = 0.0
                    prev_state = np.zeros(shape=(3,))

                    # simulate
                    START = time.time()
                    for i in range(
                        0, int(DEFAULT_DURATION_SEC * eval_env.SIM_FREQ), NUM_PHYSICS_STEPS
                    ):
                        log = {
                            "x": next_obs[0],
                            "y": next_obs[1],
                            "z": next_obs[2],
                            # "err": next_obs[-3:],
                            "err": np.linalg.norm(next_obs[-3:]),
                            "action_mag": None,
                            "xe": None,
                            "ye": None,
                            "ze": None,
                        }

                        # temp_old_state = next_obs

                        prev_obs = next_obs[:3]

                        action = load_algo.eval(next_obs, agent_weights)
                        next_obs, reward, terminated, truncated, info = eval_env.step(action)
                        distance_travelled += np.linalg.norm(next_obs[:3] - prev_state)
                        prev_state = next_obs[:3]

                        log["action_mag"] = np.linalg.norm(action[0:3])

                        log["xe"] = next_obs[0] - (prev_obs[0] + action[0])
                        log["ye"] = next_obs[1] - (prev_obs[1] + action[1])
                        log["ze"] = next_obs[2] - (prev_obs[2] + action[2])

                        coordinates.append(log)

                        if i % eval_env.SIM_FREQ == 0:
                            eval_env.render()

                        if DEFAULT_GUI:
                            sync(i, START, eval_env.TIMESTEP)

                    eval_env.reset()
                    eval_env.close()
                    del eval_env

                    df_coordinates = pd.DataFrame(coordinates)

                    df_coordinates.to_csv(os.path.join(PLT_LOGS_PATH, f"agent_{ix}", f"{dir}_{np.sum(dist):.3f}.csv"), index=False)

                json.dump(metrics, open(os.path.join(DEFAULT_OUTPUT_FOLDER, "metrics.json"), "w"), indent=4, cls=NumpyEncoder)
                print("***** dumped results")

        ### eval conditioned reward        

        # def scalarization(reward: np.ndarray):
        #         return np.linalg.norm(reward * [1, 0.5])

        # for a in algo.archive.individuals:
        #         print(eval_mo_reward_conditioned(a, env=env, scalarization=scalarization))

        ###

        ### individual policy predictions

        # # (weights - agent) pairs
        # # number of agents = pop_size param
        # for a in load_algo.agents:
        #     print(a)
        #     print(a.weights)
        #     # mo_ppo network
        #     # print(a.networks)
        #     # predict critic: networks.critic()
        #     print(a.networks.get_value(th.zeros(size=(1, env.observation_space.shape[0]))))
        #     # predict actor: networks.actor_mean()
        #     print(a.networks.get_action_and_value(th.zeros(size=(1, env.observation_space.shape[0]))))
        #     print("-----")

        ###


