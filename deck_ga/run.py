#!/usr/bin/env python3
"""
Entry point untuk Genetic Algorithm Deck Optimization.

Usage:
    python deck_ga/run.py                         # Default run
    python deck_ga/run.py --generations 100       # Custom generations
    python deck_ga/run.py --workers 4             # Parallel workers
    python deck_ga/run.py --population 200        # Larger population
    python deck_ga/run.py --games 10              # More games per evaluation
    python deck_ga/run.py --quick                 # Quick test (5 gens, small pop)
    python deck_ga/run.py --eval deck.csv         # Evaluate existing deck
"""
import os
import sys
import argparse

# Add root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm for Pokemon TCG Deck Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generations", "-g", type=int, default=None,
                        help=f"Number of generations (default: {__import__('config', fromlist=['']).NUM_GENERATIONS})")
    parser.add_argument("--population", "-p", type=int, default=None,
                        help=f"Population size (default: {__import__('config', fromlist=['']).POPULATION_SIZE})")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="Number of parallel evaluator workers")
    parser.add_argument("--games", type=int, default=None,
                        help=f"Games per evaluation (default: {__import__('config', fromlist=['']).GAMES_PER_EVAL})")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick test: 5 generations, pop=20, games=3")
    parser.add_argument("--seed-decks", "-s", type=str, default=None,
                        help="Seed GA population from folder of existing deck CSVs (e.g. agent_rl/deck_generated)")
    parser.add_argument("--eval", "-e", type=str, default=None,
                        help="Evaluate a deck CSV file against random opponents")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def run_ga(args):
    """Run full GA evolution."""
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    from . import config
    from .card_db import CardDB
    from .ga_loop import GALoop

    # Override config if provided
    if args.generations is not None:
        config.NUM_GENERATIONS = args.generations
    if args.population is not None:
        config.POPULATION_SIZE = args.population
    if args.games is not None:
        config.GAMES_PER_EVAL = args.games
    if args.quick:
        config.NUM_GENERATIONS = 5
        config.POPULATION_SIZE = 20
        config.GAMES_PER_EVAL = 3

    # Load card database
    print(f"Loading card database from {config.CARD_DB_PATH}...")
    db = CardDB(config.CARD_DB_PATH)
    print(f"Loaded {len(db)} unique cards")

    # Run GA
    ga = GALoop(db, n_workers=args.workers)
    if args.seed_decks:
        print(f"Seeding GA population from '{args.seed_decks}'...")
        ga.init_population_from_decks(args.seed_decks)
    else:
        print(f"Using random population (no --seed-decks provided).")
        ga.init_population()
    ga.run()
    return ga


def eval_deck(deck_path: str):
    """Evaluate a single deck file."""
    from . import config
    from .card_db import CardDB
    from .evaluator import DeckEvaluator

    print(f"Loading deck from {deck_path}...")
    deck = []
    with open(deck_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                deck.append(int(line))

    print(f"Deck has {len(deck)} cards")
    db = CardDB(config.CARD_DB_PATH)

    # Validate
    from .genome import DeckGenome
    genome = DeckGenome(deck, db)
    valid, errors = genome.validate()
    if not valid:
        print(f"Deck validation failed:")
        for e in errors:
            print(f"  - {e}")
        return

    print(f"Deck summary:")
    print(f"  {genome.summary()}")

    # Evaluate vs random opponents
    from .ga_loop import DeckGenome as DG
    opponents = [DG(db=db) for _ in range(3)]

    with DeckEvaluator(n_workers=1) as evaluator:
        for i, opp in enumerate(opponents):
            result = evaluator.evaluate(deck, opp.card_ids, num_games=5)
            wr = result["wins_p0"] / 5.0
            print(f"  vs random opponent {i+1}: {result['wins_p0']}/{5} ({wr:.1%})")
            print(f"    Steps: {np.mean(result['steps']):.0f}, Reasons: {result['reasons']}")


def main():
    args = parse_args()

    if args.eval:
        eval_deck(args.eval)
    else:
        run_ga(args)


if __name__ == "__main__":
    main()
