"""
Módulo de entorno para BlackjackAI.

Encapsula el entorno Blackjack-v1 de Gymnasium y expone funciones
limpias para inicializar, resetear y ejecutar pasos en el juego.
"""

import gymnasium as gym


class BlackjackEnvironment:
    """
    Wrapper sobre el entorno Blackjack-v1 de Gymnasium.

    El estado del entorno es una tupla de tres valores:
        - player_sum   (int): suma de las cartas del jugador (4–21)
        - dealer_card  (int): carta visible del dealer (1–10)
        - usable_ace   (bool): True si el jugador tiene un as que cuenta como 11

    Las acciones posibles son:
        - 0 = stick  (plantarse)
        - 1 = hit    (pedir carta)
    """

    def __init__(self, render_mode: str = None):
        """
        Inicializa el entorno Blackjack-v1.

        Args:
            render_mode: None para entrenamiento silencioso,
                         'human' para visualización en terminal (lento).
        """
        self.env = gym.make("Blackjack-v1", render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self) -> tuple:
        """
        Reinicia el entorno para un nuevo episodio.

        Returns:
            state (tuple): estado inicial (player_sum, dealer_card, usable_ace)
        """
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> tuple:
        """
        Ejecuta una acción en el entorno.

        Args:
            action (int): 0 = stick, 1 = hit

        Returns:
            next_state (tuple): nuevo estado del juego
            reward     (float): +1 victoria, 0 empate, -1 derrota
            done       (bool):  True si el episodio terminó
            info       (dict):  información adicional de Gymnasium
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info

    def close(self):
        """Cierra el entorno y libera recursos."""
        self.env.close()

    def describe_state(self, state: tuple) -> str:
        """
        Devuelve una descripción legible del estado actual.

        Args:
            state (tuple): (player_sum, dealer_card, usable_ace)

        Returns:
            str: descripción en texto plano
        """
        player_sum, dealer_card, usable_ace = state
        ace_text = "con as usable" if usable_ace else "sin as usable"
        return (
            f"Jugador: {player_sum} puntos | "
            f"Dealer muestra: {dealer_card} | "
            f"{ace_text}"
        )

    def sample_action(self) -> int:
        """
        Retorna una acción aleatoria válida (útil para pruebas).

        Returns:
            int: 0 o 1
        """
        return self.action_space.sample()


def run_episode(env: BlackjackEnvironment, agent) -> dict:
    """
    Ejecuta una partida completa usando el agente dado.

    Esta función es el corazón de la simulación: recorre el ciclo
    observar → actuar → recibir recompensa hasta que el episodio termina.

    Args:
        env   (BlackjackEnvironment): el entorno inicializado
        agent: cualquier agente que tenga un método choose_action(state)

    Returns:
        dict con:
            - reward   (float): recompensa final del episodio
            - steps    (int):   número de pasos ejecutados
            - states   (list):  estados visitados durante el episodio
            - actions  (list):  acciones tomadas en cada paso
            - result   (str):   'win', 'loss' o 'draw'
    """
    state = env.reset()
    done = False

    total_reward = 0.0
    steps = 0
    visited_states = []
    taken_actions = []

    while not done:
        visited_states.append(state)

        action = agent.choose_action(state)
        taken_actions.append(action)

        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1
        state = next_state

    result = _classify_result(total_reward)

    return {
        "reward": total_reward,
        "steps": steps,
        "states": visited_states,
        "actions": taken_actions,
        "result": result,
    }


def _classify_result(reward: float) -> str:
    """
    Convierte la recompensa final en una etiqueta legible.

    En Blackjack-v1:
        +1.0 → victoria
         0.0 → empate
        -1.0 → derrota
        +1.5 → blackjack natural (victoria especial)
    """
    if reward > 0:
        return "win"
    elif reward == 0:
        return "draw"
    else:
        return "loss"