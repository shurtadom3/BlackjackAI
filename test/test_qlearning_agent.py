# File: test_qlearning_agent.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.agents.q_learning_agent import QLearningAgent
from src.agents.random_agent import RandomAgent
from src.training.trainer import Trainer

# ===============================
# 1. Inicialización del entorno
# ===============================
env = gym.make("Blackjack-v1")  # Entorno oficial

# ===============================
# 2. Inicialización del agente Q-Learning
# ===============================
# Para Blackjack: state_space es (22, 12, 2)
# Player sum: 4-21, Dealer card: 1-11, Usable ace: True/False
agent = QLearningAgent(
    action_space=env.action_space.n,
    state_space=(22, 12, 2),
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.01
)

# ===============================
# 3. Entrenamiento del agente
# ===============================
trainer = Trainer(env, agent, episodes=1000)
episode_rewards = trainer.train()  # Entrenamiento del agente

# Guardar la Q-table entrenada
trainer.save_q_table('data/models/q_table.pkl')

# ===============================
# 4. Evaluación del agente Q-Learning
# ===============================
def evaluate_agent(env, agent, episodes=1000):
    wins, losses, draws, total_reward = 0, 0, 0, 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            episode_reward += reward

        total_reward += episode_reward
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "draw_rate": draws / episodes,
        "avg_reward": total_reward / episodes
    }

q_metrics = evaluate_agent(env, agent, episodes=1000)
print("Agente Q-Learning:", q_metrics)

# ===============================
# 5. Evaluación del agente aleatorio
# ===============================
random_agent = RandomAgent(action_space=env.action_space.n)
random_metrics = evaluate_agent(env, random_agent, episodes=1000)
print("Agente Aleatorio:", random_metrics)

# ===============================
# 6. Guardar métricas en CSV
# ===============================
df_metrics = pd.DataFrame({
    "Metric": ["Win Rate", "Loss Rate", "Draw Rate", "Average Reward"],
    "Q-Learning": [
        q_metrics["win_rate"],
        q_metrics["loss_rate"],
        q_metrics["draw_rate"],
        q_metrics["avg_reward"]
    ],
    "Random": [
        random_metrics["win_rate"],
        random_metrics["loss_rate"],
        random_metrics["draw_rate"],
        random_metrics["avg_reward"]
    ]
})
df_metrics.to_csv("data/metrics/evaluation_results.csv", index=False)
print("Métricas guardadas en data/metrics/evaluation_results.csv")

# ===============================
# 7. Visualización de resultados
# ===============================
# Comparación win/loss/draw
df_metrics.set_index("Metric")[["Q-Learning", "Random"]].plot(kind="bar", figsize=(8,5))
plt.ylabel("Proporción")
plt.title("Comparación: Agente Q-Learning vs Aleatorio")
plt.grid(axis="y")
plt.show()

# Comparación reward promedio
plt.bar(["Q-Learning", "Random"], [q_metrics["avg_reward"], random_metrics["avg_reward"]], color=["green", "red"])
plt.ylabel("Reward Promedio")
plt.title("Recompensa promedio por agente")
plt.show()

# ===============================
# 8. Recompensas por episodio (opcional)
# ===============================
plt.plot(episode_rewards)
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.title("Evolución del reward durante entrenamiento")
plt.show()