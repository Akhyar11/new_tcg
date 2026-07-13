"""
GA Loop — Main evolution loop untuk deck optimization.

Alur per generasi:
    1. Evaluasi fitness semua deck di populasi (vs random opponents + benchmark)
    2. Hitung diversity penalty
    3. Tournament selection untuk parent
    4. Crossover + mutation → offspring
    5. Elitism: pertahankan N deck terbaik
    6. Log progress (fitness, diversity, benchmark perf)
    7. Simpan checkpoint

Improvements v2:
    - Benchmark fixed decks (stabil antar generasi)
    - Diversity penalty (mencegah premature convergence)
    - Lebih banyak game per evaluasi
    - Tracking benchmark performance sebagai metrik absolut
"""
import os
import time
import random
import glob
import numpy as np
from typing import Optional

from . import config
from .card_db import CardDB
from .genome import DeckGenome, deck_cosine_similarity
from .evaluator import DeckEvaluator
from .surrogate import SurrogateModel


class GALoop:
    """
    Main GA loop.

    Args:
        db: Card database instance
        n_workers: Number of parallel evaluator workers
    """

    def __init__(self, db: CardDB, n_workers: int = 2, use_gpu: bool = False):
        self.db = db
        self.evaluator = DeckEvaluator(n_workers=n_workers, use_gpu=use_gpu)
        self.surrogate = SurrogateModel()
        self.population: list[DeckGenome] = []
        self.generation = 0
        self.best_deck: Optional[DeckGenome] = None
        self.all_best_decks: list[DeckGenome] = []

        # Benchmark decks (fixed opponents — loaded dari generated decks)
        self.benchmark_decks: list[DeckGenome] = []

        self.history = {
            "gen": [],
            "best_fitness": [],
            "avg_fitness": [],
            "diversity": [],
            "winrate_vs_benchmark": [],
            "avg_steps": [],
            "end_reasons": [],
        }

    def load_benchmark_decks(self, num_decks: int = None):
        """
        Load benchmark decks dari folder generated decks.
        Benchmark bersifat tetap — tidak berubah antar generasi.
        Ini memberikan metrik absolut yang stabil.
        """
        if num_decks is None:
            num_decks = config.NUM_BENCHMARK_OPPONENTS

        deck_dir = config.BENCHMARK_DECK_DIR
        if not os.path.exists(deck_dir):
            print(f"[GA] WARNING: Benchmark dir '{deck_dir}' not found, skipping")
            return

        files = sorted(glob.glob(os.path.join(deck_dir, "*.csv")))
        if not files:
            print(f"[GA] WARNING: No benchmark decks found in '{deck_dir}'")
            return

        # Pilih subset yang diverse untuk benchmark
        random.shuffle(files)
        loaded = 0
        for f in files:
            if loaded >= num_decks:
                break
            try:
                d = DeckGenome.from_csv(f, self.db)
                if d.is_valid():
                    self.benchmark_decks.append(d)
                    loaded += 1
            except Exception:
                pass

        print(f"[GA] Loaded {len(self.benchmark_decks)} benchmark decks for stable evaluation")

    def init_population(self, size: int = None):
        """Buat populasi awal dengan deck random."""
        if size is None:
            size = config.POPULATION_SIZE
        self.population = []
        print(f"[GA] Generating initial population of {size} decks...")
        start = time.time()
        for i in range(size):
            deck = DeckGenome(db=self.db)
            self.population.append(deck)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{size} decks generated...")
        elapsed = time.time() - start
        print(f"[GA] Population ready in {elapsed:.1f}s")
        self.generation = 0

    def init_population_from_decks(self, deck_dir, size: int = None):
        """
        Seed populasi awal dari folder deck yang sudah ada (generated decks).
        Jika jumlah deck di folder < size, isi sisanya dengan deck random.

        Args:
            deck_dir: Path ke folder berisi file .csv deck (satu ID per baris), atau list of paths.
            size: Target populasi (default: config.POPULATION_SIZE).
        """
        if size is None:
            size = config.POPULATION_SIZE
        self.population = []

        loaded = []
        if isinstance(deck_dir, str):
            deck_dir = [deck_dir]
            
        for d_dir in deck_dir:
            if len(loaded) >= size:
                break
            loaded.extend(DeckGenome.from_csv_dir(d_dir, self.db, max_count=size - len(loaded)))

        self.population.extend(loaded)
        print(f"[GA] Loaded {len(loaded)} decks from '{deck_dir}'")

        remaining = size - len(self.population)
        if remaining > 0:
            print(f"[GA] Generating {remaining} random decks to fill population...")
            for i in range(remaining):
                deck = DeckGenome(db=self.db)
                self.population.append(deck)
        else:
            self.population = self.population[:size]

        print(f"[GA] Population ready: {len(self.population)} decks (seeded from {deck_dir})")
        self.generation = 0

    # ─── Diversity ───
    def calc_population_diversity(self) -> float:
        """
        Hitung rata-rata pairwise cosine distance dalam populasi.
        0.0 = semua identik, >0 = diverse.
        """
        if len(self.population) < 2:
            return 1.0

        sample = random.sample(self.population, min(config.DIVERSITY_POOL_SIZE, len(self.population)))
        total_dist = 0.0
        count = 0
        for i in range(len(sample)):
            for j in range(i + 1, min(i + 5, len(sample))):
                sim = deck_cosine_similarity(sample[i].card_ids, sample[j].card_ids)
                total_dist += (1.0 - sim)  # distance = 1 - similarity
                count += 1
        return total_dist / max(count, 1)

    def calc_diversity_penalty(self, deck: DeckGenome) -> float:
        """
        Hitung diversity penalty untuk sebuah deck.
        Semakin mirip deck dengan populasi lain, semakin besar penalty.
        """
        if len(self.population) < 2:
            return 0.0

        sample = random.sample(
            [d for d in self.population if d is not deck],
            min(config.DIVERSITY_POOL_SIZE, len(self.population) - 1)
        )
        if not sample:
            return 0.0

        avg_sim = np.mean([deck_cosine_similarity(deck.card_ids, other.card_ids) for other in sample])
        return config.DIVERSITY_PENALTY * avg_sim

    # ─── Fitness Evaluation ───
    def evaluate_fitness(self, deck: DeckGenome, num_games: int = None):
        """
        Evaluate fitness of a single deck — SEMUA lawan dikirim CONCURRENT ke worker.

        - 3 random opponents dari populasi (diversity)
        - 2 benchmark decks (stability)

        Dengan 4 worker, 5 evaluasi hampir sepenuhnya paralel.
        """
        if num_games is None:
            num_games = config.GAMES_PER_EVAL

        # ── Kumpulkan semua opponent dan jumlah gamenya ──
        tasks: list[tuple[list[int], int]] = []  # (card_ids, num_games)

        # 1. Random opponents
        random_opponents = random.sample(self.population, min(3, len(self.population)))
        for opp in random_opponents:
            tasks.append((opp.card_ids, num_games))

        # 2. Benchmark opponents
        for bench in self.benchmark_decks:
            tasks.append((bench.card_ids, config.BENCHMARK_EVAL_GAMES))

        # ── Kirim SEMUA ke worker concurrently, lalu kumpulkan ──
        raw_results = self.evaluator.evaluate_batch_varied(deck.card_ids, tasks)

        # ── Aggregasi hasil ──
        total_wins = 0
        total_games = 0
        all_steps = []
        all_reasons = {}

        benchmark_wins = 0
        benchmark_games = 0

        for i, result in enumerate(raw_results):
            is_benchmark = i >= len(random_opponents)

            if is_benchmark:
                benchmark_wins += result["wins_p0"]
                benchmark_games += config.BENCHMARK_EVAL_GAMES
            else:
                total_wins += result["wins_p0"]
                total_games += num_games

            all_steps.extend(result["steps"])
            for r, c in result["reasons"].items():
                all_reasons[r] = all_reasons.get(r, 0) + c

        # Fitness = win rate
        total_games_effective = total_games + benchmark_games
        total_wins_effective = total_wins + benchmark_wins
        win_rate = total_wins_effective / max(total_games_effective, 1)

        hitung_diversity_penalty = self.calc_diversity_penalty(deck)
        adjusted_fitness = max(0.0, win_rate - hitung_diversity_penalty)

        avg_steps = np.mean(all_steps) if all_steps else 0

        deck.fitness = adjusted_fitness
        deck.extra_stats = {
            "raw_win_rate": win_rate,
            "diversity_penalty": hitung_diversity_penalty,
            "win_rate": win_rate,
            "benchmark_win_rate": benchmark_wins / max(benchmark_games, 1),
            "avg_steps": avg_steps,
            "total_games": total_games_effective,
            "wins": total_wins_effective,
            "reasons": all_reasons,
        }

    def evaluate_population(self):
        """Evaluate fitness of all decks in parallel."""
        total = len(self.population)
        print(f"\n[GA] Evaluating population (gen {self.generation})...")

        self.population.sort(key=lambda d: d.fitness if d.fitness is not None else 0, reverse=True)

        for i, deck in enumerate(self.population):
            if deck.fitness is not None:
                continue

            self.evaluate_fitness(deck)

            # Update surrogate with real data
            if deck.fitness is not None:
                self.surrogate.add_observation(
                    deck.card_ids, deck.fitness, self.db
                )

            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i+1}/{total}...")

        # Update best deck based on RAW win rate (before diversity penalty)
        self.population.sort(
            key=lambda d: d.extra_stats.get("raw_win_rate", 0) if d.extra_stats else 0,
            reverse=True
        )
        if self.population:
            best = self.population[0]
            best_raw = best.extra_stats.get("raw_win_rate", 0)
            current_best_raw = self.best_deck.extra_stats.get("raw_win_rate", 0) if self.best_deck else -1
            if self.best_deck is None or best_raw > current_best_raw:
                self.best_deck = best
                self.all_best_decks.append(best)
                self._save_best_deck(best)

    def tournament_selection(self, tournament_size: int = None) -> DeckGenome:
        """Select parent via tournament selection."""
        if tournament_size is None:
            tournament_size = config.TOURNAMENT_SIZE

        candidates = random.sample(self.population, min(tournament_size, len(self.population)))
        candidates.sort(key=lambda d: d.fitness if d.fitness is not None else -1, reverse=True)
        return candidates[0]

    def create_next_generation(self):
        """Create next generation via selection, crossover, mutation."""
        new_population = []

        # Elitism: keep top N (by adjusted fitness)
        self.population.sort(key=lambda d: d.fitness if d.fitness is not None else -1, reverse=True)
        for i in range(min(config.ELITISM, len(self.population))):
            elite = self.population[i]
            new_population.append(elite)

        # Fill rest via crossover + mutation
        while len(new_population) < config.POPULATION_SIZE:
            parent_a = self.tournament_selection()
            parent_b = self.tournament_selection()

            child_a, child_b = parent_a.crossover(parent_b)

            child_a.mutate()
            child_b.mutate()

            new_population.append(child_a)
            if len(new_population) < config.POPULATION_SIZE:
                new_population.append(child_b)

        # Reset fitness for re-evaluation
        for deck in new_population:
            deck.fitness = None
            deck.extra_stats = {}

        self.population = new_population[:config.POPULATION_SIZE]
        self.generation += 1

    def run(self, num_generations: int = None):
        """Run entire GA evolution."""
        if num_generations is None:
            num_generations = config.NUM_GENERATIONS

        if not self.population:
            self.init_population()

        # Load benchmark decks
        self.load_benchmark_decks()

        os.makedirs(config.DECK_OUTPUT_DIR, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  GENETIC ALGORITHM — Deck Optimization v2")
        print(f"  Population: {config.POPULATION_SIZE}")
        print(f"  Generations: {num_generations}")
        print(f"  Games/Deck vs Random: {config.GAMES_PER_EVAL}")
        print(f"  Games/Deck vs Benchmark: {config.BENCHMARK_EVAL_GAMES}")
        print(f"  Benchmark Decks: {len(self.benchmark_decks)}")
        print(f"  Workers: {len(self.evaluator.pipes)}")
        print(f"  Parallel Batch: YES (5 opponents concurrent)")
        print(f"  Diversity Penalty: {config.DIVERSITY_PENALTY}")
        print(f"  Tournament Size: {config.TOURNAMENT_SIZE}")
        print(f"  Elitism: {config.ELITISM}")
        print(f"  Mutation Rate: {config.MUTATION_RATE}")
        print(f"{'='*70}")

        for gen in range(num_generations):
            gen_start = time.time()
            self.generation = gen

            # 1. Evaluate fitness
            self.evaluate_population()

            # 2. Calculate stats
            fitnesses = [d.fitness for d in self.population if d.fitness is not None]
            raw_wrs = [d.extra_stats.get("raw_win_rate", 0) for d in self.population if d.extra_stats]

            best_fitness = max(fitnesses) if fitnesses else 0
            avg_fitness = np.mean(fitnesses) if fitnesses else 0
            best_raw = max(raw_wrs) if raw_wrs else 0
            avg_raw = np.mean(raw_wrs) if raw_wrs else 0

            # 3. Diversity
            diversity = self.calc_population_diversity()

            # 4. Benchmark performance
            bench_wrs = [
                d.extra_stats.get("benchmark_win_rate", 0)
                for d in self.population if d.extra_stats and "benchmark_win_rate" in d.extra_stats
            ]
            avg_bench_wr = np.mean(bench_wrs) if bench_wrs else 0.0

            # 5. Collect end reasons
            all_reasons = {}
            for d in self.population:
                if d.extra_stats and "reasons" in d.extra_stats:
                    for r, c in d.extra_stats["reasons"].items():
                        all_reasons[r] = all_reasons.get(r, 0) + c

            # 6. Log
            elapsed = time.time() - gen_start
            reason_labels = {1: "Prize", 2: "DeckOut", 3: "NoActive", 4: "Effect"}
            reason_str = " | ".join(
                f"{reason_labels.get(r, f'R{r}')}:{c}"
                for r, c in sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
            ) if all_reasons else "N/A"

            best_deck = self.population[0] if self.population else None
            best_line = ""
            if best_deck:
                lines = best_deck.extract_evolution_lines()
                best_line = "; ".join(
                    " → ".join(self.db.by_id(cid).name if self.db.by_id(cid) else "?" for cid in line)
                    for line in lines[:2]
                )

            print(f"\n─── Gen {gen:03d}/{num_generations} ({elapsed:.1f}s) ───")
            print(f"  Fitness (adj): {best_fitness:.4f} | Raw: {best_raw:.4f} | Avg: {avg_raw:.4f}")
            print(f"  Diversity: {diversity:.3f} | Benchmark WR: {avg_bench_wr:.3f}")
            print(f"  End Reasons: {reason_str}")
            if best_deck:
                evo = best_deck.extract_evolution_lines()
                print(f"  Best Lines: {best_line}")
            print(f"  ─────────────────────────────────────")

            # 7. Save history
            self.history["gen"].append(gen)
            self.history["best_fitness"].append(best_fitness if best_fitness else best_raw)
            self.history["avg_fitness"].append(avg_fitness if avg_fitness else avg_raw)
            self.history["diversity"].append(diversity)
            self.history["winrate_vs_benchmark"].append(avg_bench_wr)

            # 8. Train surrogate every N generations
            if config.USE_SURROGATE and gen % config.SURROGATE_TRAIN_INTERVAL == 0 and gen > 0:
                self.surrogate.train()

            # 9. Create next generation
            if gen < num_generations - 1:
                self.create_next_generation()

        # Final save
        self._save_best_deck(self.best_deck)
        self._save_history()
        self._print_summary()
        self.evaluator.close()

    def _save_best_deck(self, deck: Optional[DeckGenome]):
        """Save best deck to CSV."""
        if deck is None:
            return
        raw_wr = deck.extra_stats.get("raw_win_rate", 0) if deck.extra_stats else 0
        fname = f"best_gen_{self.generation:03d}_wr_{raw_wr:.3f}.csv"
        path = os.path.join(config.DECK_OUTPUT_DIR, fname)
        deck.to_csv(path)
        path_latest = os.path.join(config.DECK_OUTPUT_DIR, "best_current.csv")
        deck.to_csv(path_latest)

    def _save_history(self):
        """Save evolution history to CSV."""
        path = os.path.join(config.DECK_OUTPUT_DIR, "history.csv")
        with open(path, "w") as f:
            f.write("gen,best_fitness,avg_fitness,diversity,benchmark_winrate\n")
            for i in range(len(self.history["gen"])):
                gen = self.history["gen"][i]
                bf = self.history["best_fitness"][i]
                af = self.history["avg_fitness"][i]
                div = self.history["diversity"][i]
                bwr = self.history["winrate_vs_benchmark"][i]
                f.write(f"{gen},{bf:.4f},{af:.4f},{div:.4f},{bwr:.4f}\n")
        print(f"[GA] History saved to {path}")

        # Also save in JSON format for easier plotting
        import json
        json_path = os.path.join(config.DECK_OUTPUT_DIR, "history.json")
        json_data = {
            "generations": self.history["gen"],
            "best_fitness": [float(f"{v:.4f}") for v in self.history["best_fitness"]],
            "avg_fitness": [float(f"{v:.4f}") for v in self.history["avg_fitness"]],
            "diversity": [float(f"{v:.4f}") for v in self.history["diversity"]],
            "benchmark_winrate": [float(f"{v:.4f}") for v in self.history["winrate_vs_benchmark"]],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"[GA] History JSON saved to {json_path}")

    def _print_summary(self):
        """Print final summary."""
        print(f"\n{'='*70}")
        print(f"  GA COMPLETE")
        best = self.best_deck
        if best:
            raw_wr = best.extra_stats.get("raw_win_rate", 0) if best.extra_stats else 0
            print(f"  Best Raw Win Rate: {raw_wr:.4f}")
            print(f"  Best Deck: {best.summary()}")
        if self.history["diversity"]:
            avg_div = np.mean(self.history["diversity"])
            print(f"  Avg Diversity: {avg_div:.3f}")
        if self.history["winrate_vs_benchmark"]:
            final_bench = self.history["winrate_vs_benchmark"][-1]
            print(f"  Final Benchmark WR: {final_bench:.4f}")
        print(f"  Decks saved to: {config.DECK_OUTPUT_DIR}")
        print(f"{'='*70}")
