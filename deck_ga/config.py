"""
Hyperparameter Genetic Algorithm untuk Deck Optimization.
"""
import os

# ─── GA Core ───
POPULATION_SIZE = 100
NUM_GENERATIONS = 50
TOURNAMENT_SIZE = 7              # ↑ Naik dari 3 → selection pressure lebih kuat
ELITISM = 5                      # ↑ Naik dari 2 → lebih banyak elite dipertahankan
CROSSOVER_RATE = 0.85            # ↑ Naik sedikit dari 0.80
MUTATION_RATE = 0.15             # ↓ Turun dari 0.30 → konvergensi lebih stabil
GAMES_PER_EVAL = 15              # ↑ Naik dari 5 → fitness signal lebih stabil

# ─── Deck Constraints ───
DECK_SIZE = 60
MAX_SAME_NAME = 4                # Maksimal 4 kartu dengan nama yang sama
MAX_ACE_SPEC = 1                 # Maksimal 1 ACE SPEC per deck
MIN_BASIC_POKEMON = 1            # Minimum 1 Basic Pokémon (engine enforce)

# ─── Benchmark / Fixed Opponents ───
# Deck benchmark tetap yang digunakan setiap evaluasi, lintas generasi.
# Fitness terhadap benchmark = absolut, tidak berubah antar generasi.
NUM_BENCHMARK_OPPONENTS = 2      # Jumlah benchmark tetap untuk tiap evaluasi
BENCHMARK_DECK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent_rl", "deck_generated"
)

# ─── Diversity ───
# Fitness = win_rate - diversity_penalty * avg_similarity
# diversity_penalty = 0 → mati, > 0 → mendorong diversity
DIVERSITY_PENALTY = 0.05
DIVERSITY_POOL_SIZE = 20         # Ambil sample populasi untuk hitung similarity

# ─── Crossover Strategy ───
# 'line'   = preserve evolution lines + trainer core (smart)
# 'split'  = random split 30/30
# 'type'   = split by energy type
CROSSOVER_STRATEGY = 'line'

# ─── Mutation Strategy ───
# 'card_swap'       = ganti 1-3 kartu random
# 'energy_tune'     = adjust jumlah energy
# 'trainer_tune'    = ganti trainer
# 'evo_line_swap'   = ganti satu evolution line dengan line lain
# 'ratio_tune'      = adjust ratio Pokemon:Trainer:Energy
MUTATION_STRATEGIES = [
    'card_swap', 'energy_tune', 'trainer_tune',
    'evo_line_swap',
]

# ─── Fitness Weights ───
# Final fitness = win_rate * WIN_WEIGHT + avg_steps * STEP_WEIGHT + ...
# (higher weight = more important)
WIN_WEIGHT = 1.0
STEP_WEIGHT = 0.001              # Bonus kecil untuk game cepat
DAMAGE_WEIGHT = 0.0005           # Bonus kecil untuk damage output

# ─── Deck Composition Targets (untuk template-based generation) ───
# Rasio ideal untuk populasi awal: Pokemon : Trainer : Energy
TARGET_POKEMON_RANGE = (12, 16)      # Ideal: 12-16 Pokemon
TARGET_TRAINER_RANGE = (29, 38)      # Ideal: 29-38 Trainers
TARGET_ENERGY_RANGE = (10, 15)       # Ideal: 10-15 Energy
TARGET_EVO_LINES = (2, 3)            # Ideal evolution lines per deck
BENCHMARK_EVAL_GAMES = 10            # Game vs benchmark (lebih banyak untuk akurasi)

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
SURROGATE_TRAIN_INTERVAL = 5     # Train surrogate setiap N generasi
