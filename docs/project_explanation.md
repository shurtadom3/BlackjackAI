# Explicación del proyecto

## ¿Qué problema resuelve?

El proyecto trabaja sobre el problema de tomar decisiones óptimas en un entorno con incertidumbre. Usamos Blackjack como dominio porque es un juego donde el jugador tiene información parcial (no sabe qué cartas van a salir) y en cada turno tiene que decidir si sigue pidiendo o se planta.

La idea es que el agente aprenda por sí solo cuándo hacer cada cosa, sin que nadie le programe las reglas. Al final del entrenamiento el agente gana alrededor del 41% de las partidas contra el 29% del agente aleatorio, lo que muestra que efectivamente aprendió algo útil.

---

## ¿Cómo está organizada la arquitectura?

El proyecto tiene una estructura modular donde cada parte tiene una responsabilidad clara:

```
Entorno (Gymnasium Blackjack-v1)
       ↓
BlackjackEnvironment  →  wrapper limpio con reset/step/close
       ↓
Agente  →  RandomAgent o QLearningAgent
       ↓
Trainer  →  orquesta el ciclo de entrenamiento
       ↓
Evaluator  →  compara agentes corriendo N episodios
       ↓
app.py  →  interfaz que integra todo
```

El entorno no sabe nada del agente, y el agente no sabe nada del trainer. Eso hace que sea fácil cambiar una parte sin romper las demás. Por ejemplo, si quisiéramos reemplazar Q-Learning por DQN solo habría que cambiar el agente.

---

## ¿Cómo se usa la interfaz?

La interfaz está hecha con Streamlit y tiene cinco secciones:

1. **Inicio** — descripción general, arquitectura y diagrama de flujo
2. **Entrenar agente** — permite configurar los hiperparámetros y ver el entrenamiento en tiempo real con barra de progreso
3. **Evaluar agente** — corre N episodios con ambos agentes y compara sus métricas
4. **Métricas** — gráficas del win rate, reward por episodio, y un mapa de calor con la política aprendida
5. **Demo interactiva** — se puede ver al agente decidir paso a paso, o jugar uno mismo

Para lanzarla:

```bash
streamlit run app.py
```

---

## ¿Cómo se relaciona con el curso?

El proyecto cubre varios conceptos del curso de forma directa:

- El agente Q-Learning es un ejemplo concreto de **agente racional**: toma decisiones para maximizar su utilidad (la recompensa acumulada).
- El entorno Blackjack es un **MDP**: tiene estados, acciones, transiciones estocásticas y recompensas.
- La política ε-greedy es la solución clásica al problema de **exploración vs. explotación**.
- Comparamos el agente contra un baseline aleatorio para **evaluar si realmente aprendió** algo y no solo tuvo suerte.

---

## ¿Qué resultados se obtuvieron?

Con 10.000 episodios de entrenamiento:

| Agente | Win rate | Reward promedio |
|---|---|---|
| Aleatorio | ~29% | ~-0.42 |
| Q-Learning | ~41% | ~-0.17 |

La diferencia de 12 puntos porcentuales en win rate es estadísticamente significativa cuando se evalúa con 2000 partidas. Además, la política que aprende el agente tiene sentido: se planta con 17 o más puntos y pide carta con sumas bajas, que coincide con la estrategia básica del Blackjack real.

---

## ¿Qué limitaciones tiene?

**Técnicas:**
- La Q-table no generaliza. Si un estado se visitó pocas veces durante el entrenamiento, la decisión ahí puede ser mala.
- No usa experience replay, así que cada experiencia se aprovecha una sola vez.
- Con hiperparámetros mal configurados (alpha muy alto, por ejemplo) el agente puede no converger bien.

**Del dominio:**
- El entorno Blackjack-v1 es simplificado: no modela múltiples mazos, doble, split ni seguro.
- El óptimo teórico con estrategia perfecta es ~42-43% de victorias. Nuestro agente llega cerca con 10.000 episodios pero no siempre lo supera.
- En casinos reales la ventaja de la casa es estructural, ningún agente gana sistemáticamente sin contar cartas.
