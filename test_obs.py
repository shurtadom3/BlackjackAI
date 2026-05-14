import gymnasium as gym
env = gym.make("Blackjack-v1", sab=True)
print(env.observation_space)
print(getattr(env.observation_space, 'n', 'No n attribute'))
