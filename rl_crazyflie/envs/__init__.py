from gym.envs.registration import register

register(
    id="balance-aviary-v0",
    entry_point="rl_crazyflie.envs.BalanceAviary:BalanceAviary",
)