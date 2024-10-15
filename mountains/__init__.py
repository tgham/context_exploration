from gymnasium.envs.registration import register

register(
    id="mountains/GridWorld-v0",
    entry_point="mountains.envs:GridWorldEnv",
)
