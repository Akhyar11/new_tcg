"""
Hyperparameter Genetic Algorithm untuk Deck Optimization.
"""
import os

# ─── GA Core ───
POPULATION_SIZE = 100
NUM_GENERATIONS = 50
TOURNAMENT_SIZE = 3
ELITISM = 2                # Jumlah deck terbaik yang dipertahankan tiap generasi
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.30       # Probabilitas mutasi per deck (bukan per kartu)
GAMES_PER_EVAL = 5         # Jumlah game per evaluasi fitness (makin banyak makin akurat)

# ─── Deck Constraints ───
DECK_SIZE = 60
MAX_SAME_NAME = 4          # Maksimal 4 kartu dengan nama yang sama
MAX_ACE_SPEC = 1           # Maksimal 1 ACE SPEC per deck
MIN_BASIC_POKEMON = 1      # Minimum 1 Basic Pokémon (engine enforce)

# ─── Crossover Strategy ───
# 'line'   = preserve evolution lines (smart)
# 'split'  = random split 30/30
# 'type'   = split by energy type
CROSSOVER_STRATEGY = 'line'

# ─── Mutation Strategy ───
# 'card_swap'     = ganti 1-3 kartu random
# 'energy_tune'   = adjust jumlah energy
# 'trainer_tune'  = ganti trainer
MUTATION_STRATEGIES = ['card_swap', 'energy_tune', 'trainer_tune']

# ─── Fitness Weights ───
# Final fitness = win_rate * WIN_WEIGHT + avg_steps * STEP_WEIGHT + ...
# (higher weight = more important)
WIN_WEIGHT = 1.0
STEP_WEIGHT = 0.001        # Bonus kecil untuk game cepat
DAMAGE_WEIGHT = 0.0005     # Bonus kecil untuk damage output

# ─── Paths ───
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD_DB_PATH = os.path.join(ROOT_DIR, "agent_rl", "EN_Card_Data.csv")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_final.msgpack")
DECK_OUTPUT_DIR = os.path.join(ROOT_DIR, "deck_ga", "best_decks")
GENERATED_DECK_DIR = os.path.join(ROOT_DIR, "agent_rl", "deck_generated")

# ─── Self-Play Opponent ───
# 'random' = random deck dari populasi
# 'fixed'  = fixed deck
# 'best'   = best deck dari generasi sebelumnya
OPPONENT_STRATEGY = 'random'

# ─── Surrogate Model (Fase 2) ───
USE_SURROGATE = False
SURROGATE_TRAIN_INTERVAL = 5  # Train surrogate setiap N generasi
