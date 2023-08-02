import os
import pickle
import time
import gym
import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
import torch as th

from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from rl_crazyflie.envs.BalanceAviary import BalanceAviary
from rl_crazyflie.utils.Logger import Logger

# MODE = "train"
MODE = "test"

# define defaults
# DEFAULT_GUI = False
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False

DIR = "results-bal-works"

DEFAULT_OUTPUT_FOLDER = f"./{DIR}"
MODEL_PATH = f"./{DIR}/model"
ENV_PATH = f"./{DIR}/env"
LOGS_PATH = f"./{DIR}/logs"
TB_LOGS_PATH = f"./{DIR}/logs"
PLT_LOGS_PATH = f"./{DIR}/plt"

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_DURATION_SEC = 2
DEFAULT_CONTROL_FREQ_HZ = 48

INIT_XYZS = np.array([[0.0, 0.0, 1.0] for _ in range(DEFAULT_NUM_DRONES)])
INIT_RPYS = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
NUM_PHYSICS_STEPS = 1

# hyperparams
NUM_TIMESTEPS = 2e6
ACTOR_NET_ARCH = [50, 100, 500, 100, 50]
CRITIC_NET_ARCH = [50, 100, 500, 100, 50]

NUM_EVAL_EPISODES = 3

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
            "output_folder": DEFAULT_OUTPUT_FOLDER,
        },
    )
    balance_env.reset()

    if MODE == "train":
        n_actions = balance_env.action_space.shape[-1]
        mu = np.zeros(n_actions)
        sigma = 0.01 * np.ones(n_actions)

        new_logger = configure(LOGS_PATH, ["stdout", "csv", "tensorboard"])
        model = PPO(
            "MlpPolicy",
            balance_env,
            # policy_kwargs=dict(net_arch=dict(pi=ACTOR_NET_ARCH, qf=CRITIC_NET_ARCH)),
            # policy_kwargs=dict(net_arch=[50, dict(pi=ACTOR_NET_ARCH, vf=CRITIC_NET_ARCH)]),
            # use_sde=True,
            verbose=0,
            # action_noise=NormalActionNoise(mu, sigma),
            tensorboard_log=TB_LOGS_PATH,
        )

        # # test
        # # while True:
        # for _ in range(10):
        #     obs, rew, done, info = balance_env.step(np.array([0.0, 0.0, 1.0]))
        #     # print("-"*10)
        #     # print(obs)
        #     # print("-"*10)
        # exit(0)

        # resume training
        if os.path.exists(ENV_PATH) and os.path.exists(MODEL_PATH + ".zip"):
            balance_env = pickle.load(open(ENV_PATH, "rb"))
            model = PPO.load(MODEL_PATH, balance_env)


        model.set_logger(new_logger)
        model.learn(
            total_timesteps=NUM_TIMESTEPS,
            # log_interval=1,
            # callback=TBCallback(log_dir=TB_LOGS_PATH),
        )

        # save model
        model.save(MODEL_PATH)
        pickle.dump(balance_env, open(ENV_PATH, "wb"))

    elif MODE == "test":
        # balance_env = pickle.load(open(ENV_PATH, "rb"))
        model = PPO.load(MODEL_PATH, balance_env)
        # balance_env = model.get_env()

        logger = Logger(
            logging_freq_hz=int(balance_env.SIM_FREQ / balance_env.AGGR_PHY_STEPS),
            num_drones=1,
            output_folder=PLT_LOGS_PATH,
        )

        # # simulation
        # # rewards = evaluate_policy(model, balance_env, n_eval_episodes=3, return_episode_rewards=True)
        # mean_eps_reward, std_eps_reward = evaluate_policy(
        #     model, balance_env, n_eval_episodes=NUM_EVAL_EPISODES, render=False
        # )
        # mean_step_reward = mean_eps_reward / (DEFAULT_DURATION_SEC * balance_env.SIM_FREQ)

        # print(f"{mean_eps_reward=} | {std_eps_reward=} | {mean_step_reward=}")

        next_obs = balance_env.reset()

        # action, _ = model.predict(next_obs)
        # print(model.actor(next_obs)[0])

        START = time.time()
        for i in range(
            0, int(DEFAULT_DURATION_SEC * balance_env.SIM_FREQ), NUM_PHYSICS_STEPS
        ):
            action, _ = model.predict(next_obs, deterministic=True)
            next_obs, reward, done, info = balance_env.step(action)
            # print(action)

            # logger.log(
            #     drone=0,
            #     timestamp=i / balance_env.SIM_FREQ,
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
            #     balance_env.reset()

            if i % balance_env.SIM_FREQ == 0:
                balance_env.render()

            if DEFAULT_GUI:
                sync(i, START, balance_env.TIMESTEP)

if __name__ == "__main__":
    main()
