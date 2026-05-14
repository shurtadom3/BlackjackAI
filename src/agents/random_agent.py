"""
Agente aleatorio para BlackjackAI.

Este agente sirve como baseline: toma decisiones completamente al azar,
sin aprender nada. Su rendimiento establece el piso mínimo que el agente
Q-Learning debe superar para demostrar que realmente aprendió algo.
"""

import random


class RandomAgent:
    """
    Agente que selecciona acciones al azar en cada paso.

    No tiene memoria, no aprende, no tiene política. Cada acción
    es independiente del estado y de las acciones anteriores.

    Acciones posibles:
        0 = stick (plantarse)
        1 = hit   (pedir carta)
    """

    def __init__(self, seed: int = None):
        """
        Inicializa el agente aleatorio.

        Args:
            seed (int): semilla para reproducibilidad.
                        Si es None, los resultados serán distintos cada vez.
        """
        self.name = "Random Agent"
        if seed is not None:
            random.seed(seed)

    def choose_action(self, state) -> int:
        """
        Elige una acción al azar, ignorando completamente el estado.

        Args:
            state: el estado actual del entorno (no se usa)

        Returns:
            int: 0 (stick) o 1 (hit) con igual probabilidad
        """
        return random.randint(0, 1)

    def __repr__(self) -> str:
        return f"RandomAgent(name='{self.name}')"