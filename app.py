"""
app.py — Interfaz Streamlit para BlackjackAI
Estudiante 4: Interfaz, documentación e informe.
"""

import os
import pickle
import gymnasium as gym
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.agents.q_learning_agent import QLearningAgent
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="BlackjackAI — Q-Learning",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Rutas ─────────────────────────────────────────────────────────────────────
Q_TABLE_PATH = "data/models/q_table.pkl"
METRICS_PATH = "data/metrics/training_metrics.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_q_agent():
    env = gym.make("Blackjack-v1", sab=True)
    state_space = tuple(space.n for space in env.observation_space)
    env.close()
    agent = QLearningAgent(action_space=2, state_space=state_space)
    if os.path.exists(Q_TABLE_PATH):
        with open(Q_TABLE_PATH, "rb") as f:
            agent.q_table = pickle.load(f)
        agent.epsilon = 0.0
    return agent


def agent_is_trained():
    return os.path.exists(Q_TABLE_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🃏 BlackjackAI")
st.sidebar.caption("Agente Q-Learning sobre Gymnasium Blackjack-v1")

page = st.sidebar.radio(
    "Navegar",
    ["🏠 Inicio", "🎯 Entrenar agente", "📊 Evaluar agente", "📈 Métricas", "🎮 Demo interactiva"],
)

st.sidebar.markdown("---")
if agent_is_trained():
    st.sidebar.success("Agente entrenado ✅")
else:
    st.sidebar.warning("Sin agente entrenado ⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — INICIO
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Inicio":
    st.title("BlackjackAI — Agente Q-Learning")
    st.markdown(
        """
        Este proyecto implementa un agente de **aprendizaje por refuerzo** que aprende
        a jugar Blackjack usando el algoritmo **Q-Learning** sobre el entorno
        `Blackjack-v1` de [Gymnasium](https://gymnasium.farama.org/).
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Algoritmo", "Q-Learning")
    col2.metric("Entorno", "Blackjack-v1")
    col3.metric("Acciones", "Pedir (1) / Plantarse (0)")

    st.markdown("---")
    st.subheader("Arquitectura del proyecto")
    st.code(
        """
BlackjackAI/
├── src/
│   ├── environment/   # Wrapper sobre Gymnasium (Estudiante 1)
│   ├── agents/        # RandomAgent + QLearningAgent (Est. 1 y 2)
│   ├── training/      # Trainer con guardado de métricas (Est. 2 y 3)
│   ├── evaluation/    # Comparación Q-Learning vs Random (Est. 4)
│   └── visualization/ # Gráficas de entrenamiento (Est. 2 y 3)
├── data/
│   ├── models/        # Q-table entrenada (.pkl)
│   └── metrics/       # CSV con recompensas por episodio
├── app.py             # Esta interfaz (Estudiante 4)
└── docs/              # Teoría, demo y explicación del proyecto
        """,
        language="text",
    )

    st.subheader("Diagrama de flujo del sistema")
    st.markdown(
        """
        ```
        ┌──────────────┐     reset()      ┌──────────────────┐
        │   Entorno    │ ─────────────▶   │  Estado inicial  │
        │ Blackjack-v1 │                  │  (sum, dealer,   │
        └──────────────┘                  │   usable_ace)    │
               ▲                          └────────┬─────────┘
               │ step(action)                      │
               │                                   ▼
        ┌──────┴───────┐             ┌─────────────────────────┐
        │   Reward     │             │      Q-Learning Agent    │
        │ +1 / 0 / -1  │ ◀────────  │  ε-greedy → select_action│
        └──────┬───────┘             │  update_q_table(...)     │
               │                     └─────────────────────────┘
               ▼
        ┌──────────────┐
        │  ¿Terminó?   │ ──── Sí ──▶ decay_epsilon() → sig. episodio
        └──────────────┘
        ```
        """
    )

    st.subheader("Fórmula de actualización Q-Learning")
    st.latex(r"Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]")
    col1, col2 = st.columns(2)
    col1.markdown(
        """
        | Símbolo | Significado |
        |---|---|
        | $Q(s, a)$ | valor actual del par estado-acción |
        | $\\alpha$ | tasa de aprendizaje |
        | $r$ | recompensa recibida |
        """
    )
    col2.markdown(
        """
        | Símbolo | Significado |
        |---|---|
        | $\\gamma$ | factor de descuento |
        | $s'$ | estado siguiente |
        | $\\max_{a'} Q(s', a')$ | mejor valor Q en el estado siguiente |
        """
    )
    st.caption("El término entre corchetes es el error de predicción (TD error): cuánto se equivocó el agente respecto a lo esperado.")

    st.markdown("---")
    st.subheader("Relación con el curso")
    st.markdown(
        """
        | Concepto del curso | Implementación en el proyecto |
        |---|---|
        | Agente racional | `QLearningAgent` maximiza recompensa acumulada |
        | Aprendizaje por refuerzo | Q-Learning con tabla de valores Q |
        | Exploración vs explotación | Política ε-greedy con decaimiento |
        | Evaluación de rendimiento | Comparación win-rate vs agente aleatorio |
        | Entorno estocástico | Blackjack tiene aleatoriedad en las cartas |
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — ENTRENAR AGENTE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Entrenar agente":
    st.title("Entrenar el agente Q-Learning")
    st.markdown(
        "Configura los hiperparámetros y entrena el agente desde cero. "
        "El entrenamiento guarda la Q-table y las métricas automáticamente."
    )

    with st.form("train_form"):
        col1, col2 = st.columns(2)
        episodes = col1.number_input("Episodios de entrenamiento", 1000, 100_000, 10_000, 1000)
        lr = col1.slider("Tasa de aprendizaje (α)", 0.01, 1.0, 0.1, 0.01)
        discount = col2.slider("Factor de descuento (γ)", 0.5, 1.0, 0.99, 0.01)
        epsilon_decay = col2.slider("Decaimiento de ε", 0.990, 0.9999, 0.995, 0.0001, format="%.4f")
        submitted = st.form_submit_button("🚀 Iniciar entrenamiento")

    if submitted:
        env = gym.make("Blackjack-v1", sab=True)
        state_space = tuple(space.n for space in env.observation_space)
        agent = QLearningAgent(
            action_space=env.action_space.n,
            state_space=state_space,
            learning_rate=lr,
            discount_factor=discount,
            epsilon_decay=epsilon_decay,
        )
        trainer = Trainer(env, agent, episodes=int(episodes))

        progress = st.progress(0, text="Entrenando...")
        log_area = st.empty()
        logs = []

        # Patch trainer para actualizar la barra de progreso
        original_train = trainer.train

        def train_with_progress():
            wins = 0
            for episode in range(1, trainer.episodes + 1):
                state, _ = trainer.env.reset()
                total_reward = 0
                for _ in range(trainer.max_steps_per_episode):
                    action = trainer.agent.select_action(state)
                    next_state, reward, done, _, _ = trainer.env.step(action)
                    trainer.agent.update_q_table(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                    if done:
                        break
                if total_reward > 0:
                    wins += 1
                trainer.episode_rewards.append(total_reward)
                trainer.win_rates.append(wins / episode)
                trainer.agent.decay_epsilon()

                if episode % max(1, int(episodes) // 100) == 0:
                    pct = episode / int(episodes)
                    progress.progress(pct, text=f"Episodio {episode}/{int(episodes)} — Win rate: {wins/episode:.2%}")
                    if episode % max(1, int(episodes) // 10) == 0:
                        logs.append(f"Ep {episode:>6} | wr: {wins/episode:.3f} | ε: {trainer.agent.epsilon:.4f}")
                        log_area.code("\n".join(logs[-10:]))

            return trainer.episode_rewards

        train_with_progress()
        trainer.save_metrics()
        trainer.save_q_table()
        env.close()

        progress.progress(1.0, text="✅ Entrenamiento completado")
        st.success(f"Agente entrenado en {int(episodes)} episodios. Q-table guardada.")

        final_wr = trainer.win_rates[-1]
        st.metric("Win rate final", f"{final_wr:.2%}")

        # Gráfica rápida
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(trainer.win_rates, color="#2563eb", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Win rate acumulado")
        ax.set_title("Evolución del Win Rate durante el entrenamiento")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — EVALUAR AGENTE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluar agente":
    st.title("Evaluación: Q-Learning vs Agente Aleatorio")

    if not agent_is_trained():
        st.warning("Primero entrena el agente en la sección '🎯 Entrenar agente'.")
        st.stop()

    n_eval = st.slider("Episodios de evaluación", 500, 10_000, 2000, 500)

    if st.button("▶ Ejecutar evaluación"):
        with st.spinner("Evaluando ambos agentes..."):
            q_agent = load_q_agent()
            evaluator = Evaluator(n_episodes=n_eval)
            stats = evaluator.compare(q_agent)

        r = stats["random"]
        q = stats["q_learning"]

        st.subheader("Resultados")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Agente Aleatorio")
            st.metric("Win rate", f"{r['win_rate']:.2%}")
            st.metric("Draw rate", f"{r['draw_rate']:.2%}")
            st.metric("Loss rate", f"{r['loss_rate']:.2%}")
            st.metric("Reward promedio", f"{r['avg_reward']:.4f}")

        with col2:
            st.markdown("#### Agente Q-Learning")
            delta_wr = q["win_rate"] - r["win_rate"]
            st.metric("Win rate", f"{q['win_rate']:.2%}", delta=f"{delta_wr:+.2%}")
            st.metric("Draw rate", f"{q['draw_rate']:.2%}")
            st.metric("Loss rate", f"{q['loss_rate']:.2%}")
            st.metric("Reward promedio", f"{q['avg_reward']:.4f}", delta=f"{q['avg_reward']-r['avg_reward']:+.4f}")

        # Gráfica de barras comparativa
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        categories = ["Victorias", "Empates", "Derrotas"]
        r_vals = [r["win_rate"], r["draw_rate"], r["loss_rate"]]
        q_vals = [q["win_rate"], q["draw_rate"], q["loss_rate"]]
        colors_r = ["#22c55e", "#f59e0b", "#ef4444"]
        colors_q = ["#16a34a", "#d97706", "#dc2626"]

        x = np.arange(len(categories))
        width = 0.35

        axes[0].bar(x - width / 2, r_vals, width, label="Aleatorio", color=colors_r, alpha=0.7)
        axes[0].bar(x + width / 2, q_vals, width, label="Q-Learning", color=colors_q, alpha=0.95)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].set_ylabel("Proporción")
        axes[0].set_title("Comparación de resultados")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        agents = ["Aleatorio", "Q-Learning"]
        rewards = [r["avg_reward"], q["avg_reward"]]
        bar_colors = ["#94a3b8", "#2563eb"]
        axes[1].bar(agents, rewards, color=bar_colors, width=0.4)
        axes[1].set_ylabel("Reward promedio")
        axes[1].set_title("Reward promedio por episodio")
        axes[1].grid(axis="y", alpha=0.3)
        for i, v in enumerate(rewards):
            axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        improvement = (q["win_rate"] - r["win_rate"]) / max(r["win_rate"], 1e-9) * 100
        if q["win_rate"] > r["win_rate"]:
            st.success(
                f"El agente Q-Learning supera al aleatorio por **{delta_wr:.2%}** "
                f"en win rate (+{improvement:.1f}% relativo). El aprendizaje es efectivo."
            )
        else:
            st.info("El agente Q-Learning no supera al aleatorio. Considera entrenar más episodios.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — MÉTRICAS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Métricas":
    st.title("Métricas de entrenamiento")

    if not os.path.exists(METRICS_PATH):
        st.warning("No hay métricas guardadas. Entrena el agente primero.")
        st.stop()

    metrics = pd.read_csv(METRICS_PATH)
    st.caption(f"Datos de {len(metrics):,} episodios de entrenamiento")

    col1, col2, col3 = st.columns(3)
    col1.metric("Episodios totales", f"{len(metrics):,}")
    col2.metric("Win rate final", f"{metrics['win_rate'].iloc[-1]:.2%}")
    col3.metric("Reward promedio (últimos 500)", f"{metrics['reward'].iloc[-500:].mean():.4f}")

    # Win rate
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(metrics["episode"], metrics["win_rate"], color="#2563eb", linewidth=0.9, alpha=0.9)
    axes[0].set_ylabel("Win rate acumulado")
    axes[0].set_title("Evolución del Win Rate durante el entrenamiento")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=metrics["win_rate"].iloc[-1], color="red", linestyle="--", alpha=0.5,
                    label=f"Final: {metrics['win_rate'].iloc[-1]:.2%}")
    axes[0].legend()

    # Rolling reward
    window = max(50, len(metrics) // 100)
    rolling = metrics["reward"].rolling(window=window).mean()
    axes[1].plot(metrics["episode"], metrics["reward"], color="#94a3b8", linewidth=0.4, alpha=0.5, label="Reward por ep.")
    axes[1].plot(metrics["episode"], rolling, color="#f59e0b", linewidth=1.5, label=f"Media móvil ({window} ep.)")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Reward por episodio")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Q-table — Política aprendida")
    if agent_is_trained():
        q_agent = load_q_agent()
        st.markdown(
            "El siguiente mapa muestra la **acción óptima** que el agente aprendió "
            "para cada estado (sin as usable). Verde = plantarse, Rojo = pedir carta."
        )

        fig2, ax = plt.subplots(figsize=(10, 5))
        policy = np.zeros((22, 11))
        for player in range(4, 22):
            for dealer in range(1, 11):
                state_idx = (player, dealer, 0)
                policy[player, dealer] = np.argmax(q_agent.q_table[state_idx])

        im = ax.imshow(policy[4:, 1:], cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=1, origin="upper")
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(range(18))
        ax.set_yticklabels(range(4, 22))
        ax.set_xlabel("Carta visible del dealer")
        ax.set_ylabel("Suma del jugador")
        ax.set_title("Política aprendida (sin As usable)\n🟢 Plantarse (0)  🔴 Pedir carta (1)")

        green_patch = mpatches.Patch(color="#57bb8a", label="Plantarse (0)")
        red_patch = mpatches.Patch(color="#e06666", label="Pedir carta (1)")
        ax.legend(handles=[green_patch, red_patch], loc="lower right")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Tabla de métricas (primeros 20 y últimos 20 episodios)")
    preview = pd.concat([metrics.head(20), metrics.tail(20)])
    preview["win_rate"] = preview["win_rate"].apply(lambda x: f"{x:.2%}")
    st.dataframe(preview, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — DEMO INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎮 Demo interactiva":
    st.title("Demo interactiva")
    st.markdown(
        "Juega una partida de Blackjack o mira cómo decide el agente Q-Learning paso a paso."
    )

    mode = st.radio("Modo", ["🤖 Ver al agente jugar", "🙋 Jugar yo mismo"], horizontal=True)
    st.markdown("---")

    if mode == "🤖 Ver al agente jugar":
        if not agent_is_trained():
            st.warning("Primero entrena el agente.")
            st.stop()

        n_demo = st.slider("Número de partidas a simular", 1, 20, 5)
        if st.button("▶ Simular partidas"):
            q_agent = load_q_agent()
            env = gym.make("Blackjack-v1", sab=True)

            results_log = []
            for i in range(n_demo):
                state, _ = env.reset()
                done = False
                steps_log = []
                total_reward = 0

                while not done:
                    state_idx = (state[0], state[1], int(state[2]))
                    q_vals = q_agent.q_table[state_idx]
                    action = int(np.argmax(q_vals))
                    action_name = "Pedir carta 🃏" if action == 1 else "Plantarse ✋"
                    steps_log.append({
                        "Suma jugador": state[0],
                        "Dealer muestra": state[1],
                        "As usable": "Sí" if state[2] else "No",
                        "Q(stick)": f"{q_vals[0]:.4f}",
                        "Q(hit)": f"{q_vals[1]:.4f}",
                        "Decisión": action_name,
                    })
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    state = next_state

                result_label = "✅ Victoria" if total_reward > 0 else ("🟡 Empate" if total_reward == 0 else "❌ Derrota")
                results_log.append((i + 1, result_label, pd.DataFrame(steps_log)))

            env.close()

            for idx, result, df in results_log:
                with st.expander(f"Partida {idx} — {result}"):
                    st.dataframe(df, use_container_width=True)

    else:  # Jugar yo mismo
        if "game_state" not in st.session_state:
            st.session_state.game_state = None

        env_key = "demo_env"

        def start_game():
            env = gym.make("Blackjack-v1", sab=True)
            state, _ = env.reset()
            st.session_state.game_state = {
                "env": env,
                "state": state,
                "done": False,
                "history": [],
                "total_reward": 0,
            }

        def take_action(action):
            gs = st.session_state.game_state
            next_state, reward, terminated, truncated, _ = gs["env"].step(action)
            gs["history"].append({
                "Suma antes": gs["state"][0],
                "Acción": "Pedir 🃏" if action == 1 else "Plantarse ✋",
                "Reward": reward,
            })
            gs["total_reward"] += reward
            gs["state"] = next_state
            gs["done"] = terminated or truncated

        if st.button("🆕 Nueva partida"):
            start_game()

        gs = st.session_state.game_state
        if gs is not None:
            state = gs["state"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Tu suma", state[0])
            col2.metric("Dealer muestra", state[1])
            col3.metric("As usable", "Sí" if state[2] else "No")

            if not gs["done"]:
                c1, c2 = st.columns(2)
                if c1.button("🃏 Pedir carta (Hit)", use_container_width=True):
                    take_action(1)
                    st.rerun()
                if c2.button("✋ Plantarse (Stick)", use_container_width=True):
                    take_action(0)
                    st.rerun()
            else:
                r = gs["total_reward"]
                if r > 0:
                    st.success(f"¡Ganaste! Reward: {r}")
                elif r == 0:
                    st.info(f"Empate. Reward: {r}")
                else:
                    st.error(f"Perdiste. Reward: {r}")
                gs["env"].close()

            if gs["history"]:
                st.dataframe(pd.DataFrame(gs["history"]), use_container_width=True)
