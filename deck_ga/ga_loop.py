"""
GA Loop — Main evolution loop untuk deck optimization.

Alur per generasi:
    1. Evaluasi fitness semua deck di populasi
    2. Tournament selection untuk parent
    3. Crossover + mutation → offspring
    4. Elitism: pertahankan N deck terbaik
    5. Log progress
    6. Simpan checkpoint
"""
import os
import time
import random
import numpy as np
from typing import Optional

from . import config
from .card_db import CardDB
from .genome import DeckGenome
from .evaluator import DeckEvaluator
from .surrogate import SurrogateModel


class GALoop:
    """
    Main GA loop.

    Args:
        db: Card database instance
        n_workers: Number of parallel evaluator workers
    """

    def __init__(self, db: CardDB, n_workers: int = 2):
        self.db = db
        self.evaluator = DeckEvaluator(n_workers=n_workers)
        self.surrogate = SurrogateModel()
        self.population: list[DeckGenome] = []
        self.generation = 0
        self.best_deck: Optional[DeckGenome] = None
        self.all_best_decks: list[DeckGenome] = []
        self.history = {
            "gen": [],
            "best_fitness": [],
            "avg_fitness": [],
            "winrate_p0": [],
            "winrate_p1": [],
            "avg_steps": [],
            "end_reasons": [],
        }

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

    def init_population_from_decks(self, deck_dir: str, size: int = None):
        """
        Seed populasi awal dari folder deck yang sudah ada (generated decks).
        Jika jumlah deck di folder < size, isi sisanya dengan deck random.

        Args:
            deck_dir: Path ke folder berisi file .csv deck (satu ID per baris).
            size: Target populasi (default: config.POPULATION_SIZE).
        """
        if size is None:
            size = config.POPULATION_SIZE
        self.population = []

        loaded = DeckGenome.from_csv_dir(deck_dir, self.db, max_count=size)
        self.population.extend(loaded)
        print(f"[GA] Loaded {len(loaded)} decks from '{deck_dir}'")

        # Isi sisa dengan deck random
        remaining = size - len(self.population)
        if remaining > 0:
            print(f"[GA] Generating {remaining} random decks to fill population...")
            for i in range(remaining):
                deck = DeckGenome(db=self.db)
                self.population.append(deck)
        else:
            # Jika loaded melebihi size, ambil size pertama
            self.population = self.population[:size]

        print(f"[GA] Population ready: {len(self.population)} decks (seeded from {deck_dir})")
        self.generation = 0

    def evaluate_fitness(self, deck: DeckGenome, num_games: int = None):
        """Evaluate fitness of a single deck against random opponents."""
        if num_games is None:
            num_games = config.GAMES_PER_EVAL

        # Pick random opponents from population
        opponents = random.sample(self.population, min(3, len(self.population)))
        total_wins = 0
        total_games = 0
        all_steps = []
        all_reasons = {}

        for opp in opponents:
            result = self.evaluator.evaluate(deck.card_ids, opp.card_ids, num_games)
            total_wins += result["wins_p0"]
            total_games += num_games
            all_steps.extend(result["steps"])
            for r, c in result["reasons"].items():
                all_reasons[r] = all_reasons.get(r, 0) + c

        # Fitness = win rate
        win_rate = total_wins / max(total_games, 1)
        avg_steps = np.mean(all_steps) if all_steps else 0

        deck.fitness = win_rate
        deck.extra_stats = {
            "win_rate": win_rate,
            "avg_steps": avg_steps,
            "total_games": total_games,
            "wins": total_wins,
            "reasons": all_reasons,
        }

    def evaluate_population(self):
        """Evaluate fitness of all decks in parallel."""
        total = len(self.population)
        print(f"\n[GA] Evaluating population (gen {self.generation})...")

        # Evaluate top decks first (potential elites preserved)
        # Sort by fitness descending (previous gen)
        self.population.sort(key=lambda d: d.fitness if d.fitness is not None else 0, reverse=True)

        for i, deck in enumerate(self.population):
            if deck.fitness is not None:
                continue  # Skip already evaluated

            self.evaluate_fitness(deck)

            # Update surrogate with real data
            if deck.fitness is not None:
                self.surrogate.add_observation(
                    deck.card_ids, deck.fitness, self.db
                )

            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i+1}/{total}...")

        # Update best deck
        self.population.sort(key=lambda d: d.fitness if d.fitness is not None else -1, reverse=True)
        if self.population:
            best = self.population[0]
            if self.best_deck is None or best.fitness > self.best_deck.fitness:
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

        # Elitism: keep top N
        self.population.sort(key=lambda d: d.fitness if d.fitness is not None else -1, reverse=True)
        for i in range(min(config.ELITISM, len(self.population))):
            elite = self.population[i]
            new_population.append(elite)

        # Fill rest via crossover + mutation
        while len(new_population) < config.POPULATION_SIZE:
            parent_a = self.tournament_selection()
            parent_b = self.tournament_selection()

            child_a, child_b = parent_a.crossover(parent_b)

            # Mutate children
            child_a.mutate()
            child_b.mutate()

            new_population.append(child_a)
            if len(new_population) < config.POPULATION_SIZE:
                new_population.append(child_b)

        # Ensure population is clean (reset fitness for re-evaluation)
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

        os.makedirs(config.DECK_OUTPUT_DIR, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  GENETIC ALGORITHM — Deck Optimization")
        print(f"  Population: {config.POPULATION_SIZE}")
        print(f"  Generations: {num_generations}")
        print(f"  Games/Deck: {config.GAMES_PER_EVAL}")
        print(f"  Workers: {len(self.evaluator.pipes)}")
        print(f"{'='*70}")

        for gen in range(num_generations):
            gen_start = time.time()
            self.generation = gen

            # 1. Evaluate fitness
            self.evaluate_population()

            # 2. Calculate stats
            fitnesses = [d.fitness for d in self.population if d.fitness is not None]
            best_fitness = max(fitnesses) if fitnesses else 0
            avg_fitness = np.mean(fitnesses) if fitnesses else 0

            # 3. Collect end reasons across population
            all_reasons = {}
            for d in self.population:
                if d.extra_stats and "reasons" in d.extra_stats:
                    for r, c in d.extra_stats["reasons"].items():
                        all_reasons[r] = all_reasons.get(r, 0) + c

            # 4. Log
            elapsed = time.time() - gen_start
            reason_labels = {1: "Prize", 2: "DeckOut", 3: "NoActive", 4: "Effect"}
            reason_str = " | ".join(
                f"{reason_labels.get(r, f'R{r}')}:{c}"
                for r, c in sorted(all_reasons.items())
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
            print(f"  Best Fitness: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")
            print(f"  End Reasons: {reason_str}")
            if best_deck:
                evo = best_deck.extract_evolution_lines()
                print(f"  Best Lines: {best_line}")
            print(f"  ─────────────────────────────────────")

            # 5. Save history
            self.history["gen"].append(gen)
            self.history["best_fitness"].append(best_fitness)
            self.history["avg_fitness"].append(avg_fitness)

            # 6. Train surrogate every N generations
            if config.USE_SURROGATE and gen % config.SURROGATE_TRAIN_INTERVAL == 0 and gen > 0:
                self.surrogate.train()

            # 7. Create next generation
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
        fname = f"best_gen_{self.generation:03d}_fitness_{deck.fitness:.3f}.csv"
        path = os.path.join(config.DECK_OUTPUT_DIR, fname)
        deck.to_csv(path)
        # Also save as best_current.csv
        path_latest = os.path.join(config.DECK_OUTPUT_DIR, "best_current.csv")
        deck.to_csv(path_latest)

    def _save_history(self):
        """Save evolution history to CSV."""
        path = os.path.join(config.DECK_OUTPUT_DIR, "history.csv")
        with open(path, "w") as f:
            f.write("gen,best_fitness,avg_fitness\n")
            for i in range(len(self.history["gen"])):
                f.write(f"{self.history['gen'][i]},{self.history['best_fitness'][i]:.4f},{self.history['avg_fitness'][i]:.4f}\n")
        print(f"[GA] History saved to {path}")

    def _print_summary(self):
        """Print final summary."""
        print(f"\n{'='*70}")
        print(f"  GA COMPLETE")
        best = self.best_deck
        if best:
            print(f"  Best Fitness: {best.fitness:.4f}")
            print(f"  Best Deck: {best.summary()}")
        print(f"  Decks saved to: {config.DECK_OUTPUT_DIR}")
        print(f"{'='*70}")
