"""
main.py — Punto de entrada y prueba del Estudiante 1.

Ejecuta este archivo para verificar que:
  1. El entorno Blackjack-v1 se inicializa correctamente.
  2. El agente aleatorio toma acciones válidas.
  3. Una partida completa corre sin errores.
  4. Las métricas básicas se calculan bien.

Uso:
    python main.py
"""

from src.environment.blackjack_env import BlackjackEnvironment, run_episode
from src.agents.random_agent import RandomAgent


def test_environment():
    """Verifica que el entorno se inicializa y responde correctamente."""
    print("TEST 1: Inicialización del entorno")

    env = BlackjackEnvironment()
    state = env.reset()

    print(f"Estado inicial: {state}")
    print(f"Descripción:    {env.describe_state(state)}")
    print(f"Acción aleatoria de muestra: {env.sample_action()}")
    print(" Entorno inicializado correctamente\n")

    env.close()


def test_random_agent():
    """Verifica que el agente aleatorio elige acciones válidas."""
    print("TEST 2: Agente aleatorio")

    agent = RandomAgent(seed=42)
    fake_state = (15, 7, False)  # estado de ejemplo

    acciones = [agent.choose_action(fake_state) for _ in range(10)]
    print(f"10 acciones tomadas: {acciones}")

    assert all(a in (0, 1) for a in acciones), "Error: acción fuera de rango"
    print(" Todas las acciones son válidas (0 o 1)\n")


def test_full_episode():
    """Ejecuta una partida completa y muestra los resultados."""
    print("TEST 3: Partida completa")

    env = BlackjackEnvironment()
    agent = RandomAgent(seed=42)

    result = run_episode(env, agent)

    print(f"Resultado:  {result['result'].upper()}")
    print(f"Recompensa: {result['reward']}")
    print(f"Pasos:      {result['steps']}")
    print(f"Estados visitados: {result['states']}")
    print(f"Acciones tomadas:  {result['actions']}")
    print("Episodio completado sin errores\n")

    env.close()


def run_multiple_episodes(n: int = 1000):
    """
    Corre n episodios con el agente aleatorio y muestra estadísticas.
    Esto simula lo que el evaluador del Estudiante 3 hará de forma más detallada.
    """
    print(f"TEST 4: {n} episodios — estadísticas básicas")

    env = BlackjackEnvironment()
    agent = RandomAgent(seed=42)

    wins = 0
    draws = 0
    losses = 0
    total_reward = 0.0

    for _ in range(n):
        result = run_episode(env, agent)
        total_reward += result["reward"]

        if result["result"] == "win":
            wins += 1
        elif result["result"] == "draw":
            draws += 1
        else:
            losses += 1

    env.close()

    print(f"Victorias: {wins}  ({wins/n*100:.1f}%)")
    print(f"Empates:   {draws}  ({draws/n*100:.1f}%)")
    print(f"Derrotas:  {losses}  ({losses/n*100:.1f}%)")
    print(f"Reward promedio: {total_reward/n:.4f}")
    print("Estadísticas calculadas correctamente\n")


if __name__ == "__main__":
    print("\n BlackjackAI — Pruebas de Sara\n")

    test_environment()
    test_random_agent()
    test_full_episode()
    run_multiple_episodes(n=1000)

    print(" Todos los tests pasaron. El entorno y el agente aleatorio")
    print("   están listos para integrarse con el resto del proyecto.")
