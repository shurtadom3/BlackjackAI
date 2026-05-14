import numpy as np
import random

class QLearningAgent:
    def __init__(self, action_space, state_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_space = action_space  # Número de acciones posibles (0: plantarse, 1: pedir carta)
        self.state_space = state_space    # Tupla con los tamaños de las dimensiones del estado
        self.learning_rate = learning_rate  # Tasa de aprendizaje
        self.discount_factor = discount_factor  # Factor de descuento
        self.epsilon = epsilon            # Tasa de exploración
        self.epsilon_decay = epsilon_decay  # Decaimiento de epsilon
        self.min_epsilon = min_epsilon  # Límite inferior de epsilon
        
        # Inicialización de la Q-table con ceros. 
        # Como Blackjack tiene 3 dimensiones de estado, creamos un array multidimensional.
        shape = self.state_space + (self.action_space,)
        self.q_table = np.zeros(shape)
    
    def select_action(self, state):
        """ Selección de acción usando epsilon-greedy """
        # state es una tupla, e.g., (15, 7, False). Convertimos el booleano a entero (0 o 1).
        state_idx = (state[0], state[1], int(state[2]))
        
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))  # Exploración: acción aleatoria
        else:
            return np.argmax(self.q_table[state_idx])  # Explotación: mejor acción según la Q-table
    
    def update_q_table(self, state, action, reward, next_state):
        """ Actualización de la Q-table según la fórmula de Q-Learning """
        state_idx = (state[0], state[1], int(state[2]))
        next_state_idx = (next_state[0], next_state[1], int(next_state[2]))
        
        best_next_action = np.argmax(self.q_table[next_state_idx])  # Mejor acción en el siguiente estado
        
        # Fórmula de actualización de Q-Learning
        self.q_table[state_idx][action] = (1 - self.learning_rate) * self.q_table[state_idx][action] + \
            self.learning_rate * (reward + self.discount_factor * self.q_table[next_state_idx][best_next_action])
    
    def decay_epsilon(self):
        """ Reducir epsilon progresivamente """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay  # Decaimiento de epsilon
