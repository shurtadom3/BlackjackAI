"""
Módulo de evaluación para BlackjackAI.

Compara el rendimiento del agente Q-Learning contra el agente aleatorio
corriendo N episodios y calculando métricas estadísticas.
"""

import numpy as np
from src.environment.blackjack_env import BlackjackEnvironment, run_episode
from src.agents.random_agent import RandomAgent


class Evaluator:
    """
    Evalúa y compara dos agentes corriendo episodios en el entorno Blackjack.
    """

    def __init__(self, n_episodes: int = 1000, seed: int = 42):
        self.n_episodes = n_episodes
        self.seed = seed

    def _run_agent(self, agent) -> dict:
        env = BlackjackEnvironment()
        results = {"wins": 0, "draws": 0, "losses": 0, "rewards": []}

        for _ in range(self.n_episodes):
            ep = run_episode(env, agent)
            results["rewards"].append(ep["reward"])
            if ep["result"] == "win":
                results["wins"] += 1
            elif ep["result"] == "draw":
                results["draws"] += 1
            else:
                results["losses"] += 1

        env.close()

        n = self.n_episodes
        rewards = np.array(results["rewards"])
        return {
            "wins": results["wins"],
            "draws": results["draws"],
            "losses": results["losses"],
            "win_rate": results["wins"] / n,
            "draw_rate": results["draws"] / n,
            "loss_rate": results["losses"] / n,
            "avg_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "n_episodes": n,
        }

    def evaluate_random(self) -> dict:
        agent = RandomAgent(seed=self.seed)
        return self._run_agent(agent)

    def evaluate_q_agent(self, q_agent) -> dict:
        # Freeze exploration so the agent always exploits
        original_epsilon = q_agent.epsilon
        q_agent.epsilon = 0.0

        # Wrap the Q-agent so it has a choose_action interface
        class _Wrapper:
            def __init__(self, inner):
                self.inner = inner

            def choose_action(self, state):
                return self.inner.select_action(state)

        result = self._run_agent(_Wrapper(q_agent))
        q_agent.epsilon = original_epsilon
        return result

    def compare(self, q_agent) -> dict:
        random_stats = self.evaluate_random()
        q_stats = self.evaluate_q_agent(q_agent)
        return {"random": random_stats, "q_learning": q_stats}
