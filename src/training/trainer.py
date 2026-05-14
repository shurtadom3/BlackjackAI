import os
import gymnasium as gym
import numpy as np
import pickle
import pandas as pd
from src.agents.q_learning_agent import QLearningAgent

class Trainer:
    def __init__(self, env, agent, episodes=1000, max_steps_per_episode=100):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.episode_rewards = []
        self.win_rates = []

    def train(self):
        """ Entrenamiento del agente en el entorno """
        wins = 0
        for episode in range(1, self.episodes + 1):
            state, _ = self.env.reset()
            total_reward = 0
            for step in range(self.max_steps_per_episode):
                action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if done:
                    break
            
            if total_reward > 0:
                wins += 1

            self.episode_rewards.append(total_reward)
            self.win_rates.append(wins / episode)
            self.agent.decay_epsilon()
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.4f}")
        
        return self.episode_rewards

    def save_metrics(self, file_path='data/metrics/training_metrics.csv'):
        """ Guarda las métricas de entrenamiento en un archivo CSV """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        metrics = pd.DataFrame({
            'episode': range(1, self.episodes + 1),
            'reward': self.episode_rewards,
            'win_rate': self.win_rates
        })
        metrics.to_csv(file_path, index=False)

    def save_q_table(self, file_path='data/models/q_table.pkl'):
        """ Guarda la Q-table entrenada """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.agent.q_table, f)
    
    def load_q_table(self, file_path='data/models/q_table.pkl'):
        """ Carga una Q-table previamente entrenada """
        with open(file_path, "rb") as f:
            self.agent.q_table = pickle.load(f)

if __name__ == "__main__":
    # 2. Revisar el código de Q-Learning (Parte 2) y 5. Pruebas específicas
    
    # Crear el entorno de Blackjack
    env = gym.make("Blackjack-v1", sab=True)

    # El espacio de observación en Blackjack es una tupla, obtenemos los tamaños
    state_space = tuple(space.n for space in env.observation_space)

    # Crear el agente Q-Learning
    agent = QLearningAgent(action_space=env.action_space.n, state_space=state_space)

    # Crear el entrenador
    trainer = Trainer(env, agent, episodes=1000)

    # Entrenar al agente
    print("Iniciando entrenamiento...")
    rewards = trainer.train()

    # Verificar si la Q-table se actualizó (imprimimos solo una parte para no saturar)
    print("\nQ-table después del entrenamiento (estado 20, carta dealer 10, sin as):")
    print(agent.q_table[20, 10, 0])

    # Verificar que las recompensas son razonables
    print(f"\nRecompensas por episodio: {rewards[:10]}")

    # Guardar las métricas de entrenamiento (para Parte 3)
    trainer.save_metrics()
    
    # Guardar la Q-table entrenada
    trainer.save_q_table()

    print("\nEntrenamiento completado y archivos guardados.")
