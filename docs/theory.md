# Fundamentos teóricos

## Aprendizaje por refuerzo

El aprendizaje por refuerzo es un tipo de machine learning donde un agente aprende a tomar decisiones interactuando con un entorno. La idea básica es simple: el agente prueba cosas, recibe una señal de si le fue bien o mal, y ajusta su comportamiento en consecuencia. No hay un dataset etiquetado ni nadie que le diga exactamente qué hacer en cada situación.

Los componentes principales son:

| Componente | Qué es | En nuestro proyecto |
|---|---|---|
| Agente | quien toma las decisiones | `QLearningAgent` |
| Entorno | el sistema con el que interactúa | `Blackjack-v1` |
| Estado | lo que el agente observa | `(suma_jugador, carta_dealer, as_usable)` |
| Acción | lo que puede hacer | `0 = plantarse`, `1 = pedir carta` |
| Recompensa | señal de si le fue bien | `+1`, `0` o `-1` |
| Política | la estrategia que aprende | la Q-table entrenada |

---

## Proceso de Decisión de Markov (MDP)

Para que el aprendizaje por refuerzo funcione, el entorno tiene que poder modelarse como un MDP. Formalmente es la tupla `(S, A, P, R, γ)` donde S son los estados, A las acciones, P la función de transición, R la recompensa y γ el factor de descuento.

Lo importante del MDP es la **propiedad de Markov**: el siguiente estado depende únicamente del estado actual y la acción que se tomó, no de todo el historial previo. En Blackjack esto se cumple porque con saber tu suma, la carta del dealer y si tienes un as usable ya tienes toda la información relevante para decidir.

---

## Q-Learning

Q-Learning es el algoritmo que usamos para entrenar al agente. Es *model-free* (no necesita conocer la función de transición del entorno) y *off-policy* (puede aprender de experiencias generadas con una política diferente a la que está aprendiendo).

La idea es mantener una función Q(s, a) que dice cuánta recompensa total puede esperar el agente si está en el estado `s`, toma la acción `a` y sigue la política óptima después. Al principio no sabe nada, así que Q vale cero para todo. Con cada partida va actualizando esos valores.

La fórmula de actualización es:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') − Q(s, a)]
```

- `α` es la tasa de aprendizaje: cuánto cambia la estimación por cada experiencia
- `γ` es el factor de descuento: cuánto le importan las recompensas futuras vs las inmediatas
- `r + γ · max Q(s', a')` es lo que el agente cree que vale hacer esa acción en ese estado
- el término entre corchetes es básicamente "cuánto me equivoqué" — el error de predicción

Con suficientes episodios, este proceso converge a los valores Q óptimos y por lo tanto a la política óptima.

---

## Exploración vs. explotación

Uno de los problemas clásicos del RL es que el agente tiene que explorar el entorno para aprender, pero también quiere explotar lo que ya sabe para ganar. Si solo explora actúa al azar y no mejora. Si solo explota nunca descubre acciones mejores.

La solución que usamos es la política **ε-greedy**:

```
con probabilidad ε  →  acción aleatoria (exploración)
con probabilidad 1-ε → mejor acción según Q-table (explotación)
```

Al principio ε = 1.0, así que el agente actúa completamente al azar y va conociendo el entorno. Con el tiempo ε decae:

```
ε ← max(ε_min, ε × ε_decay)
```

En nuestro proyecto usamos `ε_decay = 0.995` y `ε_min = 0.01`. Eso significa que después de unos 900 episodios el agente ya casi siempre elige la mejor acción que conoce.

---

## La Q-table en Blackjack

Una de las razones por las que Blackjack es un buen caso de uso para Q-Learning es que el espacio de estados es muy pequeño:

| Dimensión | Valores posibles |
|---|---|
| Suma del jugador | 4 a 21 → 18 valores |
| Carta del dealer | 1 a 10 → 10 valores |
| As usable | True o False → 2 valores |

En total: 18 × 10 × 2 = **360 estados**. Con 2 acciones posibles, la Q-table tiene **720 entradas**. Eso cabe perfectamente en memoria y es fácil de inspeccionar.

En entornos más complejos (como Atari con imágenes como estado) esto no funcionaría y habría que usar redes neuronales para aproximar la función Q. Eso es lo que hace DQN.

---

## Relación con el curso

| Tema del curso | Cómo aparece en el proyecto |
|---|---|
| Agentes racionales | el agente actúa para maximizar su recompensa acumulada |
| Incertidumbre | el entorno es estocástico: las cartas son aleatorias |
| Aprendizaje automático | Q-Learning aprende de la experiencia sin supervisión |
| Exploración vs. explotación | política ε-greedy con decaimiento |
| Ética en IA | el agente es transparente e interpretable |

---

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3–4), 279–292.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Gymnasium. *Blackjack-v1 environment*. https://gymnasium.farama.org/environments/toy_text/blackjack/
