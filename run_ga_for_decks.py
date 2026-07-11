#!/usr/bin/env python3
"""
run_ga_for_decks.py — Jalankan GA lokal, export top N deck terbaik.

Alur:
  1. Copy model dari lokasi hasil download Kaggle → checkpoints/
  2. GA: seeded dari deck_generated/ + random fill
  3. Ambil top-N deck terbaik berdasarkan raw win rate
  4. Export ke agent_rl/ga_top_decks/ (siap upload ke Kaggle)

Usage:
    # Model dari Kaggle ada di ~/Downloads/model_final.msgpack
    python run_ga_for_decks.py --model ~/Downloads/model_final.msgpack

    # Kustom jumlah deck dan generasi
    python run_ga_for_decks.py --model model.msgpack --top-n 50 --generations 40

    # Cek hasil export
    ls agent_rl/ga_top_decks/
"""
import os
import sys
import glob
import shutil
import argparse
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GA → Export top N decks for Kaggle fine-tuning"
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path ke model_final.msgpack hasil download dari Kaggle")
    parser.add_argument("--top-n", "-n", type=int, default=50,
                        help="Jumlah deck terbaik yang diexport (default: 50)")
    parser.add_argument("--generations", "-g", type=int, default=40,
                        help="Generasi GA (default: 40)")
    parser.add_argument("--population", "-p", type=int, default=100,
                        help="Populasi GA (default: 100)")
    parser.add_argument("--games", type=int, default=8,
                        help="Games per evaluasi (default: 8)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Worker processes (default: 4)")
    parser.add_argument("--seed-decks", "-s", type=str, default=None,
                        help="Folder seed decks (default: agent_rl/deck_generated/)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: agent_rl/ga_top_decks/)")
    parser.add_argument("--gpu", action="store_true",
                        help="Gunakan GPU untuk inference (default: CPU)")
    return parser.parse_args()


def main():
    args = parse_args()
    from_deck_ga = __import__("deck_ga", fromlist=["config", "card_db", "ga_loop", "genome"])
    config = from_deck_ga.config
    CardDB = from_deck_ga.card_db.CardDB
    GALoop = from_deck_ga.ga_loop.GALoop
    DeckGenome = from_deck_ga.genome.DeckGenome

    # ── 1. Copy model ke checkpoints/ ──
    model_src = os.path.abspath(args.model)
    if not os.path.exists(model_src):
        print(f"[ERROR] Model tidak ditemukan: {model_src}")
        sys.exit(1)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    model_dst = os.path.join(config.CHECKPOINT_DIR, "model_final.msgpack")
    shutil.copy2(model_src, model_dst)
    model_size_mb = os.path.getsize(model_dst) / 1e6
    print(f"[OK] Model copied: {model_src} → {model_dst} ({model_size_mb:.1f} MB)")

    # ── 2. Tentukan seed decks ──
    seed_dir = args.seed_decks
    if seed_dir is None:
        seed_dir = os.path.join(ROOT, "agent_rl", "deck_generated")
    seed_dir = os.path.abspath(seed_dir)

    if not os.path.isdir(seed_dir):
        print(f"[WARNING] Seed dir tidak ditemukan: {seed_dir}")
        print("  Populasi akan diisi 100% random deck.")
        seed_dir = None
    else:
        n_decks = len(glob.glob(os.path.join(seed_dir, "*.csv")))
        print(f"[OK] Seed decks: {seed_dir} ({n_decks} decks)")

    # ── 3. Inisialisasi GA ──
    print(f"\n{'='*60}")
    print(f"  RUNNING GENETIC ALGORITHM")
    print(f"  Model: {model_dst}")
    print(f"  Population: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Games/Deck: {args.games}")
    print(f"  Workers: {args.workers}")
    print(f"  Target export: {args.top_n} best decks")
    print(f"{'='*60}")

    db = CardDB(config.CARD_DB_PATH)
    print(f"[OK] Card DB loaded: {len(db)} unique cards")

    ga = GALoop(db, n_workers=args.workers, use_gpu=args.gpu)

    if seed_dir and os.path.isdir(seed_dir):
        print(f"[GA] Seeding from {seed_dir}...")
        ga.init_population_from_decks(seed_dir, args.population)
    else:
        print(f"[GA] Using random population...")
        ga.init_population(args.population)

    # ── 4. Run GA ──
    t_start = time.time()
    ga.run(num_generations=args.generations)
    elapsed = time.time() - t_start
    print(f"\n[GA] Selesai dalam {elapsed/60:.1f} menit")

    # ── 5. Sort population by raw win rate ──
    ga.population.sort(
        key=lambda d: d.extra_stats.get("raw_win_rate", 0) if d.extra_stats else 0,
        reverse=True
    )

    top_decks = ga.population[:args.top_n]
    print(f"\n[Export] Top {len(top_decks)} decks (by raw win rate):")
    for i, d in enumerate(top_decks[:5]):
        wr = d.extra_stats.get("raw_win_rate", 0) if d.extra_stats else 0
        print(f"  #{i+1}: WR={wr:.3f} — {d.summary()}")
    if len(top_decks) > 5:
        print(f"  ... and {len(top_decks)-5} more")

    # ── 6. Export ke output directory ──
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(ROOT, "agent_rl", "ga_top_decks")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Hapus file lama
    old_files = glob.glob(os.path.join(out_dir, "ga_deck_*.csv"))
    for f in old_files:
        os.remove(f)

    for i, d in enumerate(top_decks):
        fname = f"ga_deck_{i:03d}.csv"
        fpath = os.path.join(out_dir, fname)
        d.to_csv(fpath)

    # Juga simpan CSV mapping deck → win rate
    mapping_path = os.path.join(out_dir, "_deck_wr_mapping.csv")
    with open(mapping_path, "w") as f:
        f.write("rank,fitness,raw_win_rate,file\n")
        for i, d in enumerate(top_decks):
            fit = d.fitness if d.fitness is not None else 0
            wr = d.extra_stats.get("raw_win_rate", 0) if d.extra_stats else 0
            fname = f"ga_deck_{i:03d}.csv"
            f.write(f"{i+1},{fit:.4f},{wr:.4f},{fname}\n")

    # ── 7. Summary ──
    # Stats distribusi deck
    all_cards = [db.by_id(cid) for d in top_decks for cid in d.card_ids]
    all_cards = [c for c in all_cards if c]
    avg_p = sum(1 for c in all_cards if c.is_pokemon) / len(top_decks)
    avg_t = sum(1 for c in all_cards if c.is_trainer) / len(top_decks)
    avg_e = sum(1 for c in all_cards if c.is_energy) / len(top_decks)

    avg_wr = sum(d.extra_stats.get("raw_win_rate", 0) if d.extra_stats else 0 for d in top_decks) / len(top_decks)
    best_wr = top_decks[0].extra_stats.get("raw_win_rate", 0) if top_decks[0].extra_stats else 0

    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"  Location: {out_dir}")
    print(f"  Decks exported: {len(top_decks)}")
    print(f"  ─────────────────────────────")
    print(f"  Best WR:     {best_wr:.3f}")
    print(f"  Avg WR:      {avg_wr:.3f}")
    print(f"  Avg Pokemon: {avg_p:.1f}")
    print(f"  Avg Trainer: {avg_t:.1f}")
    print(f"  Avg Energy:  {avg_e:.1f}")
    print(f"  ─────────────────────────────")
    print(f"  GA time:     {elapsed/60:.1f} menit")
    print(f"{'='*60}")

    print(f"\n── Next Step ──")
    print(f"  1. Upload folder '{out_dir}' ke Kaggle")
    print(f"  2. Upload model '{model_dst}' ke Kaggle (sebagai model_final.msgpack)")
    print(f"  3. Di Kaggle, jalankan:")
    print(f"     TOTAL_TIMESTEPS=2000000 \\")
    print(f"     RL_DECK_PATH=agent_rl/ga_top_decks \\")
    print(f"     FINETUNE_MODE=1 \\")
    print(f"     python agent_rl/train.py")
    print(f"")


if __name__ == "__main__":
    main()
