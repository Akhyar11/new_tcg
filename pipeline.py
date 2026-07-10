#!/usr/bin/env python3
"""
Pipeline: Alternating GA → RL Training (3 iterasi).

Asumsi:
    - Model RL sudah ditrain di Kaggle (checkpoints/model_final.msgpack sudah ada).
    - 1000 generated decks di agent_rl/deck_generated/ (seed populasi GA).

Alur:
    Iterasi 1: GA di-seed dari 1000 generated decks → cari deck optimal
               → Copy deck terbaik ke agent_rl/deck/
               → RL resume training dengan deck GA → model_v1
    Iterasi 2-3: Sama, model makin kuat → deck makin optimal → model_final

Deck Flow:
    agent_rl/deck_generated/ (1000 deck)
        ↓ seed GA population
    GA evolution (15-50 generasi)
        ↓ crossover + mutasi
    deck_ga/best_decks/ (deck optimal)
        ↓ copy_ga_decks_to_rl()
    agent_rl/deck/ (10 GA terbaik + fill random)
        ↓ load
    VectorEnv worker → battle_start(deck0, deck1)

Usage:
    python pipeline.py                          # Full pipeline (3 iterasi)
    python pipeline.py --quick                  # Quick test (1 iterasi, 5 gen GA)
    python pipeline.py --iterations 5           # 5 iterations
    python pipeline.py --resume                 # Resume dari iterasi terakhir
"""
import os
import sys
import time
import json
import shutil
import glob
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── Config ───
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
RL_DECK_DIR = os.path.join(ROOT, "agent_rl", "deck")
GA_DECK_DIR = os.path.join(ROOT, "deck_ga", "best_decks")
GENERATED_DECK_DIR = os.path.join(ROOT, "agent_rl", "deck_generated")
PIPELINE_STATE = os.path.join(ROOT, "pipeline_state.json")
DECK_BACKUP_DIR = os.path.join(ROOT, "pipeline_deck_backups")

REASON_LABELS = {1: "Prize", 2: "DeckOut", 3: "NoActive", 4: "Effect"}


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline GA → RL Training (3 iterasi)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick test: 1 iterasi, GA 5 gen, RL 100k steps")
    parser.add_argument("--iterations", "-i", type=int, default=3,
                        help="Jumlah iterasi (default: 3)")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume dari iterasi terakhir (load pipeline_state.json)")
    parser.add_argument("--ga-workers", type=int, default=2,
                        help="Worker untuk GA evaluator (default: 2)")
    parser.add_argument("--rl-workers", type=int, default=None,
                        help="NUM_ENVS untuk RL training (default: 8)")
    parser.add_argument("--ga-gens", type=int, default=None,
                        help="Generasi GA per iterasi (default: config default)")
    parser.add_argument("--rl-steps", type=int, default=None,
                        help="Total timesteps RL per iterasi (default: 300000)")
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
    return {"current_iteration": 0, "phase": "start", "completed_iterations": 0}


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


def copy_ga_decks_to_rl(n: int = 10):
    """
    Copy N deck terbaik dari GA output ke RL deck folder.
    Mengganti semua file .csv yang ada.
    """
    ga_files = sorted(glob.glob(os.path.join(GA_DECK_DIR, "*.csv")))
    if not ga_files:
        log("WARNING: Tidak ada deck hasil GA! Skip copy.")
        return 0

    # Prioritaskan best_current.csv, lalu best_gen_*.csv sorted by gen
    ga_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    # Ambil top N
    selected = ga_files[:n]
    log(f"Copying {len(selected)} GA decks to RL deck folder:")

    # Bersihin RL deck folder
    if not os.path.exists(RL_DECK_DIR):
        os.makedirs(RL_DECK_DIR)
    else:
        for f in glob.glob(os.path.join(RL_DECK_DIR, "*.csv")):
            os.remove(f)

    # Copy dengan rename terstruktur
    for i, src in enumerate(selected):
        dst = os.path.join(RL_DECK_DIR, f"ga_deck_{i:03d}.csv")
        shutil.copy2(src, dst)
        name = os.path.basename(src)
        log(f"  {i}: {name} → ga_deck_{i:03d}.csv")

    # Isi sisa dengan deck random jika kurang dari NUM_ENVS
    from deck_ga.genome import DeckGenome
    from deck_ga.card_db import CardDB

    db = CardDB(os.path.join(ROOT, "agent_rl", "EN_Card_Data.csv"))
    fill_count = max(0, 8 - len(selected))
    for i in range(fill_count):
        d = DeckGenome(db=db)
        dst = os.path.join(RL_DECK_DIR, f"random_fill_{i:03d}.csv")
        d.to_csv(dst)
    log(f"  + {fill_count} random fill decks (minimum 8 decks)")

    return len(selected)


# ─── Phase Functions ───

def phase_train_rl(iteration: int, args, total_steps: int):
    """Train RL agent — resume dari checkpoint jika ada."""
    log(f"{'='*60}")
    log(f"PHASE: RL Training — Iteration {iteration}")
    log(f"{'='*60}")

    # Modify train.py hyperparameters via env override
    os.environ["RL_TOTAL_STEPS"] = str(total_steps)
    os.environ["RL_DECK_PATH"] = RL_DECK_DIR

    # Import and run train with overrides
    # We'll call the train function with overrides
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
        log(f"RL Training: {total_steps} timesteps, resume from model_final.msgpack")
        log(f"  Deck path: {RL_DECK_DIR} ({len(glob.glob(os.path.join(RL_DECK_DIR, '*.csv')))} decks)")

        rl_train.train()

        # Rename final checkpoint to iteration-specific
        final_cp = os.path.join(CHECKPOINT_DIR, "model_final.msgpack")
        iter_cp = os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration}.msgpack")
        if os.path.exists(final_cp):
            shutil.copy2(final_cp, iter_cp)
            log(f"Checkpoint saved: model_iter_{iteration}.msgpack")

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
    ga = GALoop(db, n_workers=args.ga_workers)

    # Seed populasi dari generated decks + random fill
    if os.path.isdir(GENERATED_DECK_DIR):
        log(f"Seeding GA population from {GENERATED_DECK_DIR}...")
        ga.init_population_from_decks(GENERATED_DECK_DIR, ga_config.POPULATION_SIZE)
    else:
        log(f"Generated deck folder not found at {GENERATED_DECK_DIR}, using random population.")
        ga.init_population()
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
    log(f"  Model existing dari Kaggle → langsung loop GA → RL")
    log(f"  Iterasi: {args.iterations}")
    log(f"  Quick: {args.quick}")
    log(f"  Resume: {args.resume}")
    log(f"{'='*60}")

    # Hitung steps per RL phase
    if args.quick:
        rl_steps = 100_000
    elif args.rl_steps:
        rl_steps = args.rl_steps
    else:
        rl_steps = 300_000

    # Verifikasi model sudah ada
    model_path = os.path.join(CHECKPOINT_DIR, "model_final.msgpack")
    if not os.path.exists(model_path):
        log(f"ERROR: Model tidak ditemukan di {model_path}")
        log("Jalankan training di Kaggle dulu, atau letakkan model_final.msgpack di checkpoints/")
        return

    log(f"Model ditemukan: {model_path}")

    # State management — langsung mulai dari iteration 1, phase="ga"
    state = load_state() if args.resume else {
        "current_iteration": 1,
        "phase": "ga",
        "completed_iterations": 0,
        "start_time": time.time(),
    }

    if not args.resume:
        save_state(state)

    # ─── Iterasi 1..N: GA → RL (tanpa iterasi 0) ───
    for iteration in range(state["completed_iterations"] + 1, args.iterations + 1):
        log(f"\n{'#'*70}")
        log(f"### MAIN ITERATION {iteration}/{args.iterations}")
        log(f"{'#'*70}")

        # --- GA Phase ---
        if state["phase"] == "ga":
            log(f"\n>>> GA Phase (Iter {iteration})")
            backup_decks(iteration)
            phase_run_ga(iteration, args)
            copy_ga_decks_to_rl(args.rl_deck_samples)
            state["phase"] = "train_rl"
            save_state(state)

        # --- RL Training Phase (resume dari model_final.msgpack) ---
        if state["phase"] == "train_rl":
            log(f"\n>>> RL Training Phase (Iter {iteration}) — Resume from Kaggle model")

            deck_files = glob.glob(os.path.join(RL_DECK_DIR, "*.csv"))
            log(f"Deck folder has {len(deck_files)} decks (GA optimized)")

            if args.rl_workers:
                import agent_rl.train as rl_train
                rl_train.NUM_ENVS = args.rl_workers

            phase_train_rl(iteration, args, rl_steps)

            state["completed_iterations"] = iteration
            state["current_iteration"] = iteration
            state["phase"] = "ga"
            save_state(state)

            log(f"Iteration {iteration} selesai!")
            log(f"  RL Model: model_iter_{iteration}.msgpack")
            log(f"  GA Decks: {GA_DECK_DIR}/")

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
