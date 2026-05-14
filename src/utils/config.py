"""Configuration settings for BlackjackIA"""

# Q-Learning Parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Training Parameters
NUM_EPISODES = 10000
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Environment Parameters
DECK_SIZE = 1  # Number of decks
BLACKJACK_THRESHOLD = 21
BUST_THRESHOLD = 22

# Paths
DATA_DIR = "data"
MODELS_DIR = "data/models"
METRICS_DIR = "data/metrics"
DOCS_DIR = "docs"

# Model Files
Q_TABLE_FILE = "data/models/q_table.pkl"
METRICS_FILE = "data/metrics/training_metrics.csv"

# Random Seed for reproducibility
RANDOM_SEED = 42
