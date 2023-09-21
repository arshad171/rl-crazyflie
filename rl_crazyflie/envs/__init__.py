from gym.envs.registration import register
from gymnasium.envs.registration import register as register_gym

register(
    id="balance-aviary-v0",
    entry_point="rl_crazyflie.envs.BalanceAviary:BalanceAviary",
)

register_gym(
    id="mo-balance-aviary-v0",
    entry_point="rl_crazyflie.envs.MOBalanceAviary:MOBalanceAviary",
    max_episode_steps=1000,
)

register(
    id="navigation-aviary-v0",
    entry_point="rl_crazyflie.envs.NavigationAviary:NavigationAviary",
)

register(
    id="navigation-aviary-err-v0",
    entry_point="rl_crazyflie.envs.NavigationAviaryErr:NavigationAviaryErr",
)

register(
    id="navigation-aviary-err-u-v0",
    entry_point="rl_crazyflie.envs.NavigationAviaryErrU:NavigationAviaryErrU",
)

register_gym(
    id="mo-navigation-aviary-err-v0",
    entry_point="rl_crazyflie.envs.MONavigationAviaryErr:MONavigationAviaryErr",
    max_episode_steps=1000,
)