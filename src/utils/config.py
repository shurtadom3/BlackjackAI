# Q-Learning
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Entrenamiento
NUM_EPISODES = 10000

# Rutas
DATA_DIR = "data"
MODELS_DIR = "data/models"
METRICS_DIR = "data/metrics"
Q_TABLE_FILE = "data/models/q_table.pkl"
METRICS_FILE = "data/metrics/training_metrics.csv"

# Reproducibilidad
RANDOM_SEED = 42