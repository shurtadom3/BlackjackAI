# BlackjackAI

Proyecto final de Inteligencia Artificial. El objetivo es entrenar un agente con Q-Learning para que aprenda a jugar Blackjack solo, sin que nadie le explique la estrategia.

Usamos el entorno `Blackjack-v1` de Gymnasium, que ya implementa todas las reglas del juego. Nosotros nos encargamos del agente y de todo lo que lo rodea.

---

## ¿Qué hace el proyecto?

El agente empieza sin saber nada (la Q-table está en ceros) y aprende jugando miles de partidas. En cada partida decide si pedir carta o plantarse según lo que ha aprendido hasta ese momento. Con el tiempo mejora su estrategia y termina ganando bastante más que si jugara al azar.

El estado que ve el agente es una tupla de tres cosas:
- su suma de cartas actual
- la carta visible del dealer
- si tiene un as que cuenta como 11

Las acciones son solo dos: `0` (plantarse) y `1` (pedir carta). La recompensa es `+1` si gana, `0` si empata y `-1` si pierde.

---

## Estructura del proyecto

```
BlackjackAI/
├── src/
│   ├── environment/      # wrapper sobre Gymnasium
│   ├── agents/           # agente aleatorio y agente Q-Learning
│   ├── training/         # ciclo de entrenamiento
│   ├── evaluation/       # comparación Q-Learning vs aleatorio
│   └── visualization/    # gráficas de métricas
├── data/
│   ├── models/           # Q-table guardada en .pkl
│   └── metrics/          # CSV con métricas por episodio
├── app.py                # interfaz Streamlit
├── main.py               # tests básicos del entorno
└── docs/                 # teoría y explicación del proyecto
    ├── theory.md
    └── project_explanation.md
```

El flujo general es:

```
Entorno (reset) → estado inicial
      ↓
Agente elige acción con ε-greedy
      ↓
Entorno devuelve nuevo estado + recompensa
      ↓
Agente actualiza Q-table
      ↓
¿Terminó? → sí: siguiente episodio / no: repetir
```

---

## Cómo correrlo

Primero instalar las dependencias:

```bash
pip install -r requirements.txt
```

Para abrir la interfaz:

```bash
streamlit run app.py
```

Para correr los tests del entorno:

```bash
python main.py
```
o directamente en consola 
```bash
    python test/test_environment.py
    python test/test_random_agent.py
```

Para entrenar directamente desde terminal:

```bash
python -m src.training.trainer
```

---

## Resultados

Con 10.000 episodios de entrenamiento (configuración por defecto):

| Agente | Win rate | Reward promedio |
|---|---|---|
| Aleatorio | ~29% | ~-0.42 |
| Q-Learning | ~41% | ~-0.17 |

Mejorar un 12% sobre el aleatorio puede no parecer mucho, pero en Blackjack el óptimo teórico es ~42-43%, así que el agente está cerca del techo. La política que aprende también tiene sentido: se planta con sumas altas y pide carta con sumas bajas, que es básicamente la estrategia estándar.

---

## Limitaciones

- La Q-table no generaliza entre estados. Si algún estado se visitó poco durante el entrenamiento, la decisión ahí puede ser mala.
- No hay experience replay, así que cada experiencia se usa una sola vez.
- El entorno Blackjack-v1 es simplificado: no tiene múltiples mazos, doble ni split.
- Para entornos más grandes esto no escalaría, habría que usar redes neuronales (DQN).

---

## Ética

El proyecto es puramente educativo. No promovemos el juego de azar como forma de ganar dinero.

El agente solo ve lo mismo que vería un jugador humano normal: su propia suma y la carta visible del dealer. No tiene acceso a información oculta. Además, la Q-table es completamente interpretable, se puede ver exactamente por qué tomó cada decisión.

---

## Referencias

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Watkins & Dayan (1992). *Q-learning*. Machine Learning.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- [Gymnasium — Blackjack-v1](https://gymnasium.farama.org/environments/toy_text/blackjack/)

---

## Equipo

| Estudiante | Parte |
|---|---|
| Sara Hurtado Metaute | Entorno y agente aleatorio |
| Alyson Dahiana Henao | Agente Q-Learning y entrenamiento |
| Samuel Arango | Evaluación y visualización |
| Gabriel Atehortua| Interfaz, README y documentación |
