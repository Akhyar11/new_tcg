#!/usr/bin/env python3
"""
Pipeline: Alternating GA → RL Training (3 iterasi).

v3 — Convergence-Grade
=======================
Perbaikan untuk konvergensi training:

RL Training (train.py v3):
  - 8M timesteps default (configurable via TOTAL_TIMESTEPS env)
  - Entropy schedule: 0.05 → 0.005
  - Clip ratio schedule: 0.2 → 0.05
  - Non-symmetric opponents (P0 ≠ P1 deck) → gradient tidak cancel
  - Running reward normalization
  - Value tanh bounding [-5, +5]
  - Best model separate saving

GA (deck_ga v2):
  - Benchmark fixed opponents untuk stabilitas fitness
  - Diversity penalty (cegah premature convergence)
  - Template-based random generation (ratio realistis)
  - Evolution line-aware crossover + trainer core preservation
  - Evo line swap mutation

Alur per Iterasi:
  1. GA: seeded dari generated decks, evaluasi dengan RL agent frozen
  2. Copy N deck GA terbaik → agent_rl/deck/
  3. RL: resume training dengan deck GA sebagai opponent

Deck Assignment:
  P0 dan P1 selalu mendapat deck BERBEDA → self-play gradient
  tidak saling membatalkan. Win rate tidak stuck di 50%.

Usage:
    python pipeline.py                          # Full pipeline (3 iterasi)
    python pipeline.py --quick                  # Quick test (1 iterasi, 5 gen GA, 100k RL)
    python pipeline.py --iterations 5           # 5 iterations
    python pipeline.py --resume                 # Resume dari iterasi terakhir
    python pipeline.py --rl-steps 2000000       # 2M steps per iterasi RL
"""
import os
import sys
import time
import json
import shutil
import glob
import argparse
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# GPU Auto-Detect
# Deteksi via nvidia-smi SAJA (tanpa import JAX yang init CUDA).
# JAX hanya di-import di worker proses yang benar-benar butuh GPU.
_NUM_GPUS = 0
try:
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=5
    )
    _NUM_GPUS = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
except Exception:
    _NUM_GPUS = 0

if _NUM_GPUS > 0:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    print(f"[Pipeline] {_NUM_GPUS} GPU(s) terdeteksi - mode XLA/GPU")
    for env_key in ["CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS"]:
        val = os.environ.get(env_key, "")
        if val == "" or val.lower() in ("", "cpu"):
            os.environ.pop(env_key, None)
else:
    print("[Pipeline] GPU tidak terdeteksi - mode CPU")

# ─── Config ───
CHECKPOINT_DIR = os.environ.get("SAVE_DIR", "checkpoints")
RL_DECK_DIR = os.path.join(ROOT, "agent_rl", "deck")
GA_DECK_DIR = os.path.join(ROOT, "deck_ga", "best_decks")
GENERATED_DECK_DIR = os.path.join(ROOT, "agent_rl", "deck_generated")
PIPELINE_STATE = os.path.join(ROOT, "pipeline_state.json")
DECK_BACKUP_DIR = os.path.join(ROOT, "pipeline_deck_backups")
NEW_DECK_DIR = os.path.join(ROOT, "new_deck")

# RL defaults — cocok untuk convergence-grade training
DEFAULT_RL_STEPS = 15_000_000     # Per iterasi (3 iterasi = 45M)
QUICK_RL_STEPS = 100_000

REASON_LABELS = {1: "Prize✓", 2: "DeckOut✗", 3: "NoActive✓", 4: "Effect✓"}


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline GA → RL Training (3 iterasi)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick test: 1 iterasi, GA 5 gen, RL 100k steps")
    parser.add_argument("--iterations", "-i", type=int, default=2,
                        help="Jumlah iterasi tuning GA->RL sesudah initial train (default: 2)")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume dari iterasi terakhir (load pipeline_state.json)")
    parser.add_argument("--ga-workers", type=int, default=2,
                        help="Worker untuk GA evaluator (default: 2)")
    parser.add_argument("--rl-workers", type=int, default=None,
                        help="NUM_ENVS untuk RL training (default: 8)")
    parser.add_argument("--ga-gens", type=int, default=None,
                        help="Generasi GA per iterasi (default: config default)")
    parser.add_argument("--rl-steps", type=int, default=15000000,
                        help="Total timesteps RL per iterasi tuning (default: 15M)")
    parser.add_argument("--init-steps", type=int, default=15000000,
                        help="Total timesteps RL untuk Initial Train (default: 15M)")
    parser.add_argument("--rl-deck-samples", type=int, default=10,
                        help="Jumlah deck GA terbaik yang dicopy ke RL (default: 10)")
    return parser.parse_args()


def log(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] [Pipeline] {msg}")


def save_state(state: dict):
    with open(PIPELINE_STATE, "w") as f:
        json.dump(state, f, indent=2)
    log(f"State saved: iteration={state['current_iteration']}, phase={state['phase']}")


def load_state() -> dict:
    if os.path.exists(PIPELINE_STATE):
        with open(PIPELINE_STATE) as f:
            return json.load(f)
    return {"current_iteration": 0, "phase": "init_train", "completed_iterations": -1}


def backup_decks(iteration: int):
    """Backup deck folder sebelum diganti GA decks."""
    os.makedirs(DECK_BACKUP_DIR, exist_ok=True)
    backup_path = os.path.join(DECK_BACKUP_DIR, f"decks_iter_{iteration}")
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    if os.path.exists(RL_DECK_DIR):
        shutil.copytree(RL_DECK_DIR, backup_path)
        log(f"Deck backup → {backup_path}")


def restore_deck_backup(iteration: int):
    """Restore deck dari backup."""
    backup_path = os.path.join(DECK_BACKUP_DIR, f"decks_iter_{iteration}")
    if os.path.exists(backup_path):
        if os.path.exists(RL_DECK_DIR):
            shutil.rmtree(RL_DECK_DIR)
        shutil.copytree(backup_path, RL_DECK_DIR)
        log(f"Deck restored from {backup_path}")


# prepare_tuning_decks removed because vector_env now handles the 70/30 selection dynamically.


# ─── Phase Functions ───

def phase_train_rl(iteration: int, args, total_steps: int, new_deck_dir: str, gen_deck_dir: str):
    """Train RL agent — resume dari checkpoint jika ada."""
    log(f"{'='*60}")
    log(f"PHASE: RL Training — Iteration {iteration}")
    log(f"{'='*60}")

    # Modify train.py hyperparameters via env override
    os.environ["TOTAL_TIMESTEPS"] = str(total_steps)    # ← dibaca train.py v3
    os.environ["NEW_DECK_PATH"] = new_deck_dir
    os.environ["GEN_DECK_PATH"] = gen_deck_dir

    # GPU/CPU-aware scaling: naikkan NUM_ENVS dan BATCH_SIZE untuk memaksimalkan resource
    if _NUM_GPUS >= 1:
        # Jika menggunakan instance seperti Vast.ai (1 GPU, 12 Core, 16GB VRAM)
        # 32 envs sangat cocok untuk 12 core CPU, Batch 1024 sangat ringan untuk 16GB VRAM
        recommended_envs = 32 * _NUM_GPUS
        recommended_batch = 1024 * _NUM_GPUS
        os.environ["RL_NUM_ENVS"] = str(recommended_envs)
        os.environ["RL_BATCH_SIZE"] = str(recommended_batch)
        log(f"Resource Auto-Scaling: NUM_ENVS={recommended_envs}, BATCH_SIZE={recommended_batch} (GPU x{_NUM_GPUS})")
    elif args.rl_workers:
        os.environ["RL_NUM_ENVS"] = str(args.rl_workers)

    # Pastikan GPU visibility untuk JAX
    if _NUM_GPUS > 0:
        for env_key in ["CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS"]:
            val = os.environ.get(env_key, "")
            if val == "" or val.lower() in ("", "cpu"):
                os.environ.pop(env_key, None)

    # Import and run train with overrides
    sys.path.insert(0, ROOT)

    # Override agar menggunakan deck yang sudah di-set
    import agent_rl.train as rl_train

    # Save original values
    orig_total = rl_train.TOTAL_TIMESTEPS

    try:
        # Override
        rl_train.TOTAL_TIMESTEPS = total_steps
        rl_train.SAVE_DIR = CHECKPOINT_DIR

        # Ensure checkpoint path for save
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # Run
        log(f"RL Training: {total_steps} timesteps, resume from model_final.msgpack (if exists)")
        log(f"  New Deck path (70%): {new_deck_dir} ({len(glob.glob(os.path.join(new_deck_dir, '*.csv')))} decks)")
        log(f"  Gen Deck path (30%): {gen_deck_dir} ({len(glob.glob(os.path.join(gen_deck_dir, '*.csv')))} decks)")

        rl_train.train()

    finally:
        rl_train.TOTAL_TIMESTEPS = orig_total

    log(f"RL Training Iteration {iteration} selesai.")
    return True


def phase_run_ga(iteration: int, args):
    """Run GA dengan frozen RL model dari iterasi sebelumnya."""
    log(f"{'='*60}")
    log(f"PHASE: Genetic Algorithm — Iteration {iteration}")
    log(f"{'='*60}")

    # Ensure GA uses the latest model checkpoint
    model_path = os.path.join(CHECKPOINT_DIR, "model_final.msgpack")
    if not os.path.exists(model_path):
        log(f"WARNING: Model checkpoint tidak ditemukan di {model_path}")
        log("GA akan menggunakan random weights (tidak ideal)")

    from deck_ga.card_db import CardDB
    from deck_ga.ga_loop import GALoop
    from deck_ga import config as ga_config

    # Override GA config untuk pipeline
    if args.quick:
        ga_config.NUM_GENERATIONS = 5
        ga_config.POPULATION_SIZE = 20
        ga_config.GAMES_PER_EVAL = 3
    elif args.ga_gens:
        ga_config.NUM_GENERATIONS = args.ga_gens
    else:
        ga_config.NUM_GENERATIONS = 15  # Default lebih ringan untuk pipeline
        ga_config.POPULATION_SIZE = 80
        ga_config.GAMES_PER_EVAL = 5

    ga_config.DECK_OUTPUT_DIR = GA_DECK_DIR
    ga_config.MODEL_PATH = model_path

    db = CardDB(os.path.join(ROOT, "agent_rl", "EN_Card_Data.csv"))
    # GA Workers selalu CPU. Forward pass kecil (93×31 → 250 logits).
    # Bottleneck adalah C++ engine simulation, bukan model inference.
    # GPU disimpan untuk RL Training yang butuh throughput tinggi.
    ga = GALoop(db, n_workers=args.ga_workers, use_gpu=False)
    log(f"GA Workers menggunakan CPU (GPU reserved untuk RL training)")

    # Seed populasi:
    # Iterasi 1: dari deck awal (GENERATED_DECK_DIR)
    # Iterasi 2+: dari deck terbaik iterasi sebelumnya (RL_DECK_DIR) + deck awal
    seed_dirs = [GENERATED_DECK_DIR]
    if iteration > 1 and os.path.exists(RL_DECK_DIR) and len(os.listdir(RL_DECK_DIR)) > 0:
        seed_dirs.insert(0, RL_DECK_DIR)  # Prioritaskan RL_DECK_DIR terlebih dahulu

    log(f"Seeding GA population from {seed_dirs}...")
    ga.init_population_from_decks(seed_dirs, ga_config.POPULATION_SIZE)
    ga.run()

    # Log hasil GA
    if ga.best_deck:
        log(f"GA Best Fitness: {ga.best_deck.fitness:.4f}")
        lines = ga.best_deck.extract_evolution_lines()
        db_local = db
        line_str = "; ".join(
            " → ".join(db_local.by_id(cid).name if db_local.by_id(cid) else "?" for cid in line)
            for line in lines
        )
        log(f"GA Best Lines: {line_str}")
    else:
        log("WARNING: GA tidak menghasilkan deck valid!")

    return True


# ─── Main Pipeline ───

def run_pipeline(args):
    log(f"{'='*60}")
    log(f"  PIPELINE GA ↔ RL TRAINING")
    log(f"  Initial Train 5M steps → Loop (GA → Tuning 2M steps)")
    log(f"  Iterasi Tuning: {args.iterations}")
    log(f"  Quick: {args.quick}")
    log(f"  Resume: {args.resume}")
    log(f"{'='*60}")

    if args.quick:
        rl_steps = QUICK_RL_STEPS
        init_steps = QUICK_RL_STEPS
    else:
        rl_steps = args.rl_steps
        init_steps = args.init_steps

    # State management
    state = load_state() if args.resume else {
        "current_iteration": 0,
        "phase": "init_train",
        "completed_iterations": -1,
        "start_time": time.time(),
    }

    if not args.resume:
        save_state(state)
        
    # --- Initial RL Training (Iter 0) ---
    if state["phase"] == "init_train":
        log(f"\n>>> Initial RL Training Phase (Iter 0) - {init_steps} steps")
        
        log("Memasukkan New Decks (70%) + Random Generated Decks (30%) untuk P0 & P1 pool (Dynamic Python Choice).")
        phase_train_rl(0, args, init_steps, new_deck_dir=NEW_DECK_DIR, gen_deck_dir=GENERATED_DECK_DIR)
            
        state["phase"] = "train_rl"
        state["completed_iterations"] = 0
        state["current_iteration"] = 1
        save_state(state)

    # ─── Iterasi 1..N: RL Tuning (No GA) ───
    for iteration in range(state["completed_iterations"] + 1, args.iterations + 1):
        log(f"\n{'#'*70}")
        log(f"### TUNING ITERATION {iteration}/{args.iterations}")
        log(f"{'#'*70}")

        if state["phase"] == "ga":
            # GA is disabled, skip to train_rl
            state["phase"] = "train_rl"

        # --- RL Training Phase (resume dari model_final.msgpack) ---
        if state["phase"] == "train_rl":
            log(f"\n>>> RL Tuning Phase (Iter {iteration}) — {rl_steps} steps")

            log(f"Dynamic Python 70/30 Deck Selection Activated (from {NEW_DECK_DIR} and {GENERATED_DECK_DIR})")
            phase_train_rl(iteration, args, rl_steps, new_deck_dir=NEW_DECK_DIR, gen_deck_dir=GENERATED_DECK_DIR)

            state["completed_iterations"] = iteration
            state["current_iteration"] = iteration
            state["phase"] = "train_rl"
            save_state(state)

            log(f"Iteration {iteration} selesai!")
            log(f"  RL Model: model_final.msgpack")

    # ─── Final ───
    elapsed = time.time() - state.get("start_time", time.time())
    log(f"\n{'='*70}")
    log(f"  PIPELINE COMPLETE!")
    log(f"  Iterasi selesai: {args.iterations}")
    log(f"  Waktu total: {elapsed/60:.1f} menit")
    log(f"  Checkpoints: {CHECKPOINT_DIR}/")
    log(f"  Best decks: {GA_DECK_DIR}/")
    log(f"  Model final: model_final.msgpack (siap eval)")
    log(f"{'='*70}")

    # Cleanup pipeline state
    if os.path.exists(PIPELINE_STATE):
        os.remove(PIPELINE_STATE)
        log("Pipeline state cleaned.")


def main():
    args = parse_args()

    # Ensure directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GA_DECK_DIR, exist_ok=True)
    os.makedirs(RL_DECK_DIR, exist_ok=True)

    run_pipeline(args)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
