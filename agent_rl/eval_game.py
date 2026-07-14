#!/usr/bin/env python3
"""
Evaluation Script — Pokémon TCG RL Agent.

Menjalankan banyak game otomatis dan menghasilkan laporan statistik:
- Win rate keseluruhan & per-deck matchup
- Distribusi end reason (Prize, DeckOut, NoActive, Timeout)
- Metrik gameplay (avg steps, evolve rate, KO rate, dsb.)
- Analisis kualitas strategis

Usage:
    python agent_rl/eval_game.py                    # 100 game, default
    python agent_rl/eval_game.py --games 200        # 200 game
    python agent_rl/eval_game.py --checkpoint checkpoints/model_best.msgpack
    python agent_rl/eval_game.py --verbose           # Cetak detail per-game
    python agent_rl/eval_game.py --watch              # Mode interaktif (1 game, detail penuh)
"""
import os
import sys
import glob
import random
import time
import argparse
import csv
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force CPU untuk evaluasi — GPU tidak perlu
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from flax import serialization

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation, OptionType, LogType, AreaType
from agent_rl.feature_extractor import extract_features
from agent_rl.action_mapping import get_action_index_for_option, create_action_mask
from agent_rl.model import PokemonAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DECK_DIR = os.path.join(ROOT, "new_deck")
DEFAULT_CHECKPOINT = os.path.join(ROOT, "checkpoints", "model_final.msgpack")

REASON_LABELS = {1: "Prize", 2: "DeckOut", 3: "NoActive", 4: "Effect", 9: "Timeout"}
MAX_GAME_STEPS = 300


def load_deck(filepath):
    """Load deck dari CSV."""
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                deck.append(int(line))
    if len(deck) != 60:
        return None
    return deck


def load_card_db():
    """Load card database untuk nama kartu."""
    cards = {}
    csv_path = os.path.join(ROOT, "agent_rl", "EN_Card_Data.csv")
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cards[int(row['Card ID'])] = row
    return cards


def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / (exp_x.sum() + 1e-10)


def ai_select(model_apply, params, obs, use_argmax=False):
    """AI memilih aksi. Return (choices, info_dict)."""
    if not obs.select or not obs.select.option:
        return [], {}

    your_index = obs.current.yourIndex
    features = extract_features(obs.current, obs.select, your_index)
    seq_input = np.expand_dims(features["seq_input"], axis=0)
    glob_input = np.expand_dims(features["glob_input"], axis=0)

    logits_raw, value = model_apply(params, seq_input, glob_input)
    logits_np = np.array(logits_raw[0])
    value_np = float(np.array(value).flatten()[0])

    # Build mask
    options = obs.select.option
    min_c = obs.select.minCount
    mock_select = {
        "options": [{"type": OptionType(o.type).name, "index": o.index} for o in options]
    }
    mask_array = create_action_mask(mock_select)

    # Mask logits
    masked = logits_np - 1e9 * (1.0 - mask_array)
    probs = softmax(masked)

    # Sample atau argmax
    sampled_indices = []
    if use_argmax:
        # Argmax: pilih aksi dengan probabilitas tertinggi
        sorted_idx = np.argsort(-probs)
        for idx in sorted_idx:
            if mask_array[idx] > 0 and len(sampled_indices) < min_c:
                sampled_indices.append(int(idx))
    else:
        # Stochastic sampling (sama seperti training)
        remaining = probs.copy()
        for _ in range(min_c):
            if remaining.sum() <= 0:
                break
            p = remaining / remaining.sum()
            idx = int(np.random.choice(len(p), p=p))
            sampled_indices.append(idx)
            remaining[idx] = 0.0

    # Map ke C++ options
    choices = []
    for jax_idx in sampled_indices:
        for cpp_idx, opt in enumerate(mock_select["options"]):
            mapped_idx = get_action_index_for_option(opt)
            if mapped_idx == jax_idx and cpp_idx not in choices:
                choices.append(cpp_idx)
                break

    # Fallback
    if len(choices) < min_c:
        for cpp_idx in range(len(options)):
            if cpp_idx not in choices:
                choices.append(cpp_idx)
            if len(choices) >= min_c:
                break

    # Info untuk analisis
    top_probs = sorted(
        [(i, float(probs[i])) for i in range(250) if mask_array[i] > 0],
        key=lambda x: -x[1]
    )[:5]

    info = {
        "value": value_np,
        "top_probs": top_probs,
        "chosen_jax": sampled_indices,
        "entropy": float(-np.sum(probs * np.log(probs + 1e-10) * mask_array)),
        "n_legal": int(np.sum(mask_array)),
    }
    return choices, info


def run_single_game(model_apply, params_p0, params_p1, deck0, deck1, cards_db,
                    verbose=False, use_argmax=False):
    """Jalankan satu game. Return dict statistik."""

    try:
        obs_dict, _ = battle_start(deck0, deck1)
        obs = to_dataclass(obs_dict, Observation)
    except Exception as e:
        return {"error": str(e)}

    stats = {
        "steps": 0,
        "result": -1,
        "end_reason": 0,
        "winner": -1,
        # Tracking per-player
        "p0_attacks": 0, "p1_attacks": 0,
        "p0_evolves": 0, "p1_evolves": 0,
        "p0_kos": 0, "p1_kos": 0,
        "p0_supporters": 0, "p1_supporters": 0,
        "p0_items": 0, "p1_items": 0,
        "p0_energy_attach": 0, "p1_energy_attach": 0,
        "p0_prizes_taken": 0, "p1_prizes_taken": 0,
        "values": [],   # Value estimates sepanjang game
        "entropies": [],
    }

    old_prizes = [0, 0]  # Track prize counts

    while obs.current is not None and obs.current.result == -1:
        stats["steps"] += 1
        if stats["steps"] >= MAX_GAME_STEPS:
            stats["end_reason"] = 9
            break

        your_index = obs.current.yourIndex
        p_key = f"p{your_index}_"

        # Analisis logs
        if obs.logs:
            for log in obs.logs:
                pi = log.playerIndex if log.playerIndex is not None else -1
                pk = f"p{pi}_"
                if log.type == LogType.ATTACK:
                    stats[pk + "attacks"] = stats.get(pk + "attacks", 0) + 1
                elif log.type == LogType.EVOLVE:
                    stats[pk + "evolves"] = stats.get(pk + "evolves", 0) + 1
                elif log.type == LogType.PLAY:
                    # Cek tipe kartu dari DB
                    if log.cardId and log.cardId in cards_db:
                        ct = cards_db[log.cardId].get("Stage (Pokémon)/Type (Energy and Trainer)", "")
                        if ct == "Supporter":  # SUPPORTER
                            stats[pk + "supporters"] = stats.get(pk + "supporters", 0) + 1
                        elif ct == "Item":  # ITEM
                            stats[pk + "items"] = stats.get(pk + "items", 0) + 1
                elif log.type == LogType.ATTACH:
                    stats[pk + "energy_attach"] = stats.get(pk + "energy_attach", 0) + 1
                elif log.type == LogType.RESULT:
                    stats["result"] = log.result if log.result is not None else -1
                    stats["end_reason"] = log.reason if log.reason is not None else 0

        # Track prizes
        for pi in range(2):
            cur_prizes = len(obs.current.players[pi].prize)
            if old_prizes[pi] == 0 and cur_prizes > 0:
                old_prizes[pi] = cur_prizes
            elif cur_prizes < old_prizes[pi] and old_prizes[pi] > 0:
                taken = old_prizes[pi] - cur_prizes
                stats[f"p{pi}_prizes_taken"] += taken
                stats[f"p{pi}_kos"] += 1
                old_prizes[pi] = cur_prizes

        # AI memilih
        current_params = params_p0 if your_index == 0 else params_p1
        choices, info = ai_select(model_apply, current_params, obs, use_argmax)

        stats["values"].append(info.get("value", 0))
        stats["entropies"].append(info.get("entropy", 0))

        if verbose:
            # Board state
            state = obs.current
            for pi in range(2):
                player = state.players[pi]
                active_name = "Kosong"
                if player.active and player.active[0]:
                    pkm = player.active[0]
                    cname = cards_db.get(pkm.id, {}).get('Card Name', f'ID:{pkm.id}')
                    active_name = f"{cname} HP:{pkm.hp}/{pkm.maxHp} E:{len(pkm.energies)}"
                bench_names = []
                for b in player.bench:
                    bn = cards_db.get(b.id, {}).get('Card Name', f'ID:{b.id}')
                    bench_names.append(f"{bn}({b.hp})")
                print(f"  P{pi}: Active={active_name} | Bench=[{', '.join(bench_names)}] | "
                      f"Prize:{len(player.prize)} Deck:{player.deckCount}")

            # Aksi yang dipilih
            opt_names = []
            for c in choices:
                if obs.select and obs.select.option and c < len(obs.select.option):
                    o = obs.select.option[c]
                    opt_names.append(f"{OptionType(o.type).name}:{o.index}")
            print(f"  Step {stats['steps']} P{your_index} → {opt_names} "
                  f"(V={info.get('value', 0):+.2f} H={info.get('entropy', 0):.2f})")

        # Eksekusi aksi
        try:
            obs_dict = battle_select(choices)
            obs = to_dataclass(obs_dict, Observation)
        except Exception as e:
            try:
                opt_count = len(obs.select.option) if obs.select and obs.select.option else 0
                min_c = obs.select.minCount if obs.select else 0
                fallback = list(range(min(opt_count, min_c)))
                obs_dict = battle_select(fallback)
                obs = to_dataclass(obs_dict, Observation)
            except:
                stats["error"] = str(e)
                break

    # Final result
    if obs.current and stats["result"] == -1:
        stats["result"] = obs.current.result
        if stats["end_reason"] == 0:
            # Cek logs terakhir
            for log in obs.logs:
                if log.type == LogType.RESULT:
                    stats["end_reason"] = log.reason if log.reason is not None else 0

    if stats["result"] == 0:
        stats["winner"] = 0
    elif stats["result"] == 1:
        stats["winner"] = 1
    elif stats["result"] == 2:
        stats["winner"] = -1  # Draw

    try:
        battle_finish()
    except:
        pass
    return stats


def print_report(all_stats, deck_names_p0, deck_names_p1, elapsed):
    """Cetak laporan statistik lengkap."""
    total = len(all_stats)
    errors = sum(1 for s in all_stats if "error" in s)
    valid = [s for s in all_stats if "error" not in s]
    n = len(valid)

    if n == 0:
        print("Semua game error!")
        return

    wins_p0 = sum(1 for s in valid if s["winner"] == 0)
    wins_p1 = sum(1 for s in valid if s["winner"] == 1)
    draws = sum(1 for s in valid if s["winner"] == -1 and s["result"] == 2)
    timeouts = sum(1 for s in valid if s["end_reason"] == 9)

    print(f"\n{'='*70}")
    print(f"  📊 EVALUATION REPORT — {n} games ({elapsed:.1f}s)")
    print(f"{'='*70}")

    # ── Win Rate ──
    wr = wins_p0 / n * 100
    print(f"\n┌─── Win Rate ─────────────────────────────────────────┐")
    print(f"│  P0 (AI): {wins_p0:3d}/{n} = {wr:5.1f}%", end="")
    bar_len = 30
    filled = int(wr / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}]  │")
    print(f"│  P1 (AI): {wins_p1:3d}/{n} = {wins_p1/n*100:5.1f}%", end="")
    print(f"{'':>34s}│")
    if draws > 0:
        print(f"│  Draw:    {draws:3d}/{n} = {draws/n*100:5.1f}%", end="")
        print(f"{'':>34s}│")
    if errors > 0:
        print(f"│  Errors:  {errors:3d}/{total}", end="")
        print(f"{'':>41s}│")
    print(f"└──────────────────────────────────────────────────────┘")

    # ── End Reasons ──
    reasons = defaultdict(int)
    for s in valid:
        reasons[s["end_reason"]] += 1

    print(f"\n┌─── End Reasons ──────────────────────────────────────┐")
    for reason, count in sorted(reasons.items()):
        label = REASON_LABELS.get(reason, f"Unknown({reason})")
        pct = count / n * 100
        bar = "█" * int(pct / 100 * 20)
        print(f"│  {label:12s}: {count:3d} ({pct:5.1f}%)  {bar:20s}   │")
    print(f"└──────────────────────────────────────────────────────┘")

    # ── Gameplay Metrics ──
    avg_steps = np.mean([s["steps"] for s in valid])
    avg_p0_attacks = np.mean([s["p0_attacks"] for s in valid])
    avg_p0_evolves = np.mean([s["p0_evolves"] for s in valid])
    avg_p0_kos = np.mean([s["p0_kos"] for s in valid])
    avg_p0_supporters = np.mean([s["p0_supporters"] for s in valid])
    avg_p0_energy = np.mean([s["p0_energy_attach"] for s in valid])
    avg_p0_prizes = np.mean([s["p0_prizes_taken"] for s in valid])

    print(f"\n┌─── Gameplay Metrics (P0 avg per game) ───────────────┐")
    print(f"│  Game Length:     {avg_steps:6.1f} steps                      │")
    print(f"│  Attacks:         {avg_p0_attacks:6.2f}                            │")
    print(f"│  Evolves:         {avg_p0_evolves:6.2f}                            │")
    print(f"│  KOs:             {avg_p0_kos:6.2f}                            │")
    print(f"│  Prizes Taken:    {avg_p0_prizes:6.2f}                            │")
    print(f"│  Supporters:      {avg_p0_supporters:6.2f}                            │")
    print(f"│  Energy Attaches: {avg_p0_energy:6.2f}                            │")
    print(f"└──────────────────────────────────────────────────────┘")

    # ── Value & Entropy ──
    all_values = [v for s in valid for v in s.get("values", [])]
    all_entropies = [e for s in valid for e in s.get("entropies", [])]

    if all_values:
        print(f"\n┌─── Model Confidence ─────────────────────────────────┐")
        print(f"│  Avg Value Estimate: {np.mean(all_values):+6.3f}  (ideal: > 0)        │")
        print(f"│  Avg Entropy:        {np.mean(all_entropies):6.3f}   (low=confident)    │")
        # Value di game yang menang vs kalah
        win_vals = [v for s in valid if s["winner"] == 0 for v in s["values"]]
        lose_vals = [v for s in valid if s["winner"] == 1 for v in s["values"]]
        if win_vals:
            print(f"│  Value saat MENANG:  {np.mean(win_vals):+6.3f}                       │")
        if lose_vals:
            print(f"│  Value saat KALAH:   {np.mean(lose_vals):+6.3f}                       │")
        print(f"└──────────────────────────────────────────────────────┘")

    # ── Per-Deck Matchup ──
    matchup = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0})
    for i, s in enumerate(valid):
        d0 = deck_names_p0[i] if i < len(deck_names_p0) else "?"
        d1 = deck_names_p1[i] if i < len(deck_names_p1) else "?"
        key = f"{d0} vs {d1}"
        if s["winner"] == 0:
            matchup[key]["wins"] += 1
        elif s["winner"] == 1:
            matchup[key]["losses"] += 1
        else:
            matchup[key]["draws"] += 1

    if len(matchup) <= 30:  # Jangan cetak kalau terlalu banyak
        print(f"\n┌─── Matchup Details ──────────────────────────────────┐")
        for key, m in sorted(matchup.items(), key=lambda x: -x[1]["wins"]):
            total_m = m["wins"] + m["losses"] + m["draws"]
            wr_m = m["wins"] / total_m * 100 if total_m > 0 else 0
            # Truncate key if too long
            display_key = key[:44] if len(key) > 44 else key
            print(f"│  {display_key:44s} {m['wins']}W/{m['losses']}L {wr_m:5.1f}% │")
        print(f"└──────────────────────────────────────────────────────┘")

    # ── Diagnosis ──
    print(f"\n┌─── 🔍 Diagnosis ─────────────────────────────────────┐")
    if 40 <= wr <= 60:
        print(f"│  ⚖️  Win rate seimbang ({wr:.0f}%). Ini wajar karena P0 dan  │")
        print(f"│     P1 menggunakan model yang sama (Self-Play).      │")
    elif wr > 60:
        print(f"│  ✅ Win rate tinggi ({wr:.0f}%). Model P0 menang lebih     │")
        print(f"│     sering (mungkin faktor First-Turn Advantage).    │")
    else:
        print(f"│  ⚠️  Win rate rendah ({wr:.0f}%). Ada bias yang membuat    │")
        print(f"│     P0 lebih sering kalah.                           │")

    if avg_p0_attacks < 1.0 or avg_p0_energy < 1.0:
        print(f"│  ❌ Perhatian: Agent pasif!                          │")
        if avg_p0_attacks < 1.0:
            print(f"│     → Jarang menyerang ({avg_p0_attacks:.1f}/game)                   │")
        if avg_p0_energy < 1.0:
            print(f"│     → Jarang pasang energi ({avg_p0_energy:.1f}/game)                 │")

    reason_pct = {k: v/n*100 for k, v in reasons.items()}
    if reason_pct.get(2, 0) > 30:
        print(f"│  ⚠️  DeckOut terlalu tinggi ({reason_pct[2]:.0f}%). Agent mungkin│")
        print(f"│     stalling atau tidak agresif.                     │")
    if reason_pct.get(9, 0) > 10:
        print(f"│  ⚠️  Timeout tinggi ({reason_pct[9]:.0f}%). Game terlalu lama. │")
    print(f"└──────────────────────────────────────────────────────┘")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TCG AI Agent")
    parser.add_argument("--games", "-g", type=int, default=100,
                        help="Jumlah game yang dijalankan (default: 100)")
    parser.add_argument("--checkpoint-p0", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path ke model checkpoint untuk P0")
    parser.add_argument("--checkpoint-p1", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path ke model checkpoint untuk P1")
    parser.add_argument("--deck-dir", "-d", type=str, default=DEFAULT_DECK_DIR,
                        help="Direktori deck CSV")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Cetak detail setiap game")
    parser.add_argument("--watch", "-w", action="store_true",
                        help="Mode nonton: 1 game, detail penuh, step-by-step")
    parser.add_argument("--argmax", action="store_true",
                        help="Gunakan argmax (greedy) alih-alih sampling")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed untuk reproduksi")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.watch:
        args.games = 1
        args.verbose = True

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load decks
    deck_files = sorted(glob.glob(os.path.join(args.deck_dir, "*.csv")))
    if not deck_files:
        print(f"Tidak ada deck CSV di {args.deck_dir}!")
        return

    loaded_decks = []
    deck_basenames = []
    for f in deck_files:
        d = load_deck(f)
        if d:
            loaded_decks.append(d)
            deck_basenames.append(os.path.splitext(os.path.basename(f))[0])

    print(f"Loaded {len(loaded_decks)} decks dari {args.deck_dir}")

    # Load model
    print(f"Loading model P0: {args.checkpoint_p0}")
    print(f"Loading model P1: {args.checkpoint_p1}")
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    _, init_rng = jax.random.split(rng)

    dummy_seq = jnp.zeros((1, 113, 31))
    dummy_glob = jnp.zeros((1, 266))
    params_p0 = model.init(init_rng, dummy_seq, dummy_glob)
    params_p1 = model.init(init_rng, dummy_seq, dummy_glob)

    if os.path.exists(args.checkpoint_p0):
        with open(args.checkpoint_p0, 'rb') as f:
            params_p0 = serialization.from_bytes(params_p0, f.read())
        print(f"✅ Checkpoint P0 loaded: {args.checkpoint_p0}")
    else:
        print(f"⚠️  Checkpoint P0 tidak ditemukan! Menggunakan random weights.")

    if os.path.exists(args.checkpoint_p1):
        with open(args.checkpoint_p1, 'rb') as f:
            params_p1 = serialization.from_bytes(params_p1, f.read())
        print(f"✅ Checkpoint P1 loaded: {args.checkpoint_p1}")
    else:
        print(f"⚠️  Checkpoint P1 tidak ditemukan! Menggunakan random weights.")

    model_apply = jax.jit(model.apply)

    # Warmup JIT
    _ = model_apply(params_p0, dummy_seq, dummy_glob)
    _ = model_apply(params_p1, dummy_seq, dummy_glob)

    cards_db = load_card_db()

    # Run games
    mode = "argmax" if args.argmax else "sampling"
    print(f"\nRunning {args.games} games (mode: {mode})...\n")

    all_stats = []
    deck_names_p0 = []
    deck_names_p1 = []
    start_time = time.time()

    for game_i in range(args.games):
        idx0 = random.randint(0, len(loaded_decks) - 1)
        idx1 = random.randint(0, len(loaded_decks) - 1)

        deck_names_p0.append(deck_basenames[idx0])
        deck_names_p1.append(deck_basenames[idx1])

        if args.verbose:
            print(f"\n{'─'*60}")
            print(f"Game {game_i+1}/{args.games}: "
                  f"{deck_basenames[idx0]} vs {deck_basenames[idx1]}")
            print(f"{'─'*60}")

        stats = run_single_game(
            model_apply, params_p0, params_p1,
            loaded_decks[idx0], loaded_decks[idx1],
            cards_db,
            verbose=args.verbose,
            use_argmax=args.argmax
        )
        all_stats.append(stats)

        if args.verbose:
            winner = stats.get("winner", -1)
            reason = REASON_LABELS.get(stats.get("end_reason", 0), "?")
            print(f"  → Winner: P{winner} ({reason}) in {stats['steps']} steps")

        # Progress bar (non-verbose)
        if not args.verbose and (game_i + 1) % 10 == 0:
            wins_so_far = sum(1 for s in all_stats if s.get("winner") == 0)
            wr = wins_so_far / len(all_stats) * 100
            print(f"  [{game_i+1:3d}/{args.games}] Win P0: {wr:.1f}%")

        if args.watch:
            time.sleep(0.05)

    elapsed = time.time() - start_time
    print_report(all_stats, deck_names_p0, deck_names_p1, elapsed)


if __name__ == "__main__":
    main()
