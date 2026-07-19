#!/usr/bin/env python3
"""
Pipeline orchestrator:
Phase 1: Run train_new.py with TRAIN_PHASE=1 (LSTM vs FF) until 65% winrate over 200 window is achieved.
Phase 2: Run train_new.py with TRAIN_PHASE=2 (LSTM vs LSTM) with 60% winrate target over 150 window.
"""
import os
import sys
import subprocess
import argparse
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline LSTM Training (Phase 1 -> Phase 2)")
    parser.add_argument("--rl-steps", type=int, default=20000000,
                        help="Total timesteps per run (default: 20M)")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of environments (default: 8)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--phase", type=int, default=None,
                        help="Force run a specific phase (1 or 2)")
    return parser.parse_args()

def log(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] [PipelineNew] {msg}")

def run_train_phase(phase: int, args):
    log(f"Starting Phase {phase} training...")
    
    # Setup environment overrides
    env = os.environ.copy()
    env["TRAIN_PHASE"] = str(phase)
    env["TOTAL_TIMESTEPS"] = str(args.rl_steps)
    env["RL_NUM_ENVS"] = str(args.num_envs)
    env["RL_BATCH_SIZE"] = str(args.batch_size)
    env["SAVE_DIR"] = os.path.join(ROOT, "tcg_models")
    env["NEW_DECK_PATH"] = os.path.join(ROOT, "new_deck")
    env["GEN_DECK_PATH"] = os.path.join(ROOT, "deck_generated")
    
    cmd = [sys.executable, os.path.join(ROOT, "agent_rl_lstm", "train_new.py")]
    
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read output in real-time
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()
        
    process.wait()
    
    if process.returncode == 0:
        log(f"Phase {phase} selesai dengan sukses!")
        return True
    else:
        log(f"ERROR: Phase {phase} gagal dengan exit code {process.returncode}!")
        return False

def main():
    args = parse_args()
    
    log("="*60)
    log("  PIPELINE TRAINING BARU (LSTM VS FF -> LSTM VS LSTM)")
    log(f"  Target Timesteps: {args.rl_steps:,}")
    log(f"  Num Envs: {args.num_envs} | Batch Size: {args.batch_size}")
    log("="*60)
    
    if args.phase is not None:
        # Run a single forced phase
        success = run_train_phase(args.phase, args)
        if not success:
            sys.exit(1)
    else:
        # Run Phase 1
        success = run_train_phase(1, args)
        if not success:
            log("Pipeline dihentikan karena Phase 1 gagal.")
            sys.exit(1)
            
        # Run Phase 2
        success = run_train_phase(2, args)
        if not success:
            log("Pipeline dihentikan karena Phase 2 gagal.")
            sys.exit(1)
            
    log("="*60)
    log("  PIPELINE SELESAI DENGAN SUKSES!")
    log("="*60)

if __name__ == "__main__":
    main()
