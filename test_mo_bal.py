# %%
import numpy as np
import torch as th

import gym
import mo_gymnasium as mo_gym

from gym_pybullet_drones.utils.enums import DroneModel, Physics

from morl_baselines.common.evaluation import eval_mo, eval_mo_reward_conditioned
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.single_policy.ser.mo_ppo import make_env
from morl_baselines.common.utils import make_gif

from rl_crazyflie.envs.MOBalanceAviary import MOBalanceAviary
from rl_crazyflie.utils.constants import Modes


# %%
DIR = "results-mo"

MODEL_PATH = f"./{DIR}/model"
ENV_PATH = f"./{DIR}/env"
LOGS_PATH = f"./{DIR}/logs"
TB_LOGS_PATH = f"./{DIR}/logs"
PLT_LOGS_PATH = f"./{DIR}/plt/it"

# define defaults
VIEW = False
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

INIT_XYZS = np.array([[0.0, 0.0, 1.0] for _ in range(DEFAULT_NUM_DRONES)])
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

if __name__ == "__main__":
    env_id = "mo-balance-aviary-v0"
    ref_point = np.array([-1.0, -1.0])

    eval_env = mo_gym.make(
    env_id,
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
            "gui": False,
        },
    )

    # number of agents = pop_size (population size) param, (weights - agent) pairs
    algo = PGMORL(
        env_id=env_id,
        origin=ref_point,
        project_name="mo-bal",
        num_envs=1,
        # pop_size=1,
        warmup_iterations=1,
        evolutionary_iterations=1,
        # num_weight_candidates=7,
    )

    pf = algo.train(
        total_timesteps=int(1),
        ref_point=ref_point,
        known_pareto_front=None,
        eval_env=eval_env,
    )

    print(pf)

    # env = make_env(env_id, 422, 1, "PGMORL_test", gamma=0.995)()  # idx != 0 to avoid taking videos

    # # Execution of trained policies
    # for a in algo.archive.individuals:
    #     scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
    #         agent=a, env=env, w=np.array([1.0, 1.0]), render=True
    #     )
    #     print(f"Agent #{a.id}")
    #     print(f"Scalarized: {scalarized}")
    #     print(f"Discounted scalarized: {discounted_scalarized}")
    #     print(f"Vectorial: {reward}")
    #     print(f"Discounted vectorial: {discounted_reward}")


# %%
env = mo_gym.make(
    env_id,
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
        "gui": DEFAULT_USER_DEBUG_GUI,
    },
)

# %%
for a in algo.archive.individuals:
        # w -> weight vector for discounted reward
        scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
            agent=a, env=env, w=np.array([1.0, 1.0]), render=True
        )
        print(f"Agent #{a.id}")
        print(f"Scalarized: {scalarized}")
        print(f"Discounted scalarized: {discounted_scalarized}")
        print(f"Vectorial: {reward}")
        print(f"Discounted vectorial: {discounted_reward}")
        print("-----")

# %%
def scalarization(reward: np.ndarray):
        # return min(reward[0], reward[1] // 2)
        return np.linalg.norm(reward * [1, 0.5])

for a in algo.archive.individuals:
        print(eval_mo_reward_conditioned(a, env=env, scalarization=scalarization))

# %%
# for a in algo.archive.individuals:
#     make_gif(env, a, weight=[1, 1], fullpath="./test.gif")

# %%
# (weights - agent) pairs
# number of agents = pop_size param
for a in algo.agents:
    print(a)
    print(a.weights)
    # mo_ppo network
    print(a.networks)
    # predict critic: networks.critic()
    print(a.networks.get_value(th.zeros(size=(1, env.observation_space.shape[0]))))
    # predict actor: networks.actor_mean()
    print(a.networks.get_action_and_value(th.zeros(size=(1, env.observation_space.shape[0]))))
    print("-----")


