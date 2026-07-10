"""
Deck Evaluator — Mengevaluasi deck menggunakan frozen RL agent + C++ engine.

Arsitektur:
    EvaluatorManager (main process)
        ├── WorkerProcess #1 (C++ engine + model)
        ├── WorkerProcess #2 (C++ engine + model)
        └── ... (n_workers)
        Tiap worker: load model sekali → terima (deck_a, deck_b) → game → return result

Ini memisahkan C++ state tiap worker sehingga tidak ada race condition.
"""
import os
import sys
import multiprocessing as mp
import time
import numpy as np

# Root path for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _worker_main(pipe, worker_id):
    """
    Worker process:
    - Load model + JIT compile
    - Receive (deck_a, deck_b, num_games) tuples via pipe
    - Return results
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax
    import jax.numpy as jnp
    from flax import serialization
    import numpy as np

    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_dataclass, Observation, OptionType
    from agent_rl.feature_extractor import extract_features
    from agent_rl.action_mapping import get_action_index_for_option, create_action_mask
    from agent_rl.model import PokemonAgent

    # ─── Load Model ───
    try:
        model = PokemonAgent(num_actions=250)
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        jnp_seq = jnp.zeros((1, 93, 31))
        jnp_glob = jnp.zeros((1, 266))
        params = model.init(init_rng, jnp_seq, jnp_glob)

        cp_path = os.path.join(ROOT, "checkpoints", "model_final.msgpack")
        if os.path.exists(cp_path):
            with open(cp_path, "rb") as f:
                params = serialization.from_bytes(params, f.read())
            model_loaded = True
            model_apply = jax.jit(model.apply)
            print(f"  [Worker {worker_id}] Model loaded: {cp_path}")
        else:
            model_loaded = False
            model_apply = None
            print(f"  [Worker {worker_id}] WARNING: No checkpoint, using random weights!")
            model_apply = jax.jit(model.apply)
    except Exception as e:
        print(f"  [Worker {worker_id}] ERROR loading model: {e}")
        model_loaded = False
        model_apply = None

    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / (exp_x.sum() + 1e-10)

    def run_game(deck_a, deck_b, max_steps=500) -> dict:
        """Run a single game between deck_a (P0) and deck_b (P1)."""
        obs_dict, start_data = battle_start(deck_a, deck_b)
        if obs_dict is None:
            return {"winner": -1, "reason": "invalid_deck", "steps": 0}

        obs = to_dataclass(obs_dict, Observation)
        steps = 0

        while obs.current and obs.current.result == -1 and steps < max_steps:
            steps += 1

            your_index = obs.current.yourIndex

            if obs.select is None or not obs.select.option:
                break

            opts = obs.select.option
            min_c = obs.select.minCount

            # Build select dict for action mapping
            mock_select_dict = {
                "options": [
                    {"type": OptionType(o.type).name, "index": o.index}
                    for o in opts
                ]
            }

            # AI inference (frozen model)
            if model_apply is not None:
                features = extract_features(obs.current, obs.select, your_index)
                seq_input = np.expand_dims(features["seq_input"], axis=0)
                glob_input = np.expand_dims(features["glob_input"], axis=0)

                masked_logits, _ = model_apply(params, seq_input, glob_input)
                logits_np = np.array(masked_logits[0])

                # Categorical sampling without replacement
                mask_array = create_action_mask(mock_select_dict)
                masked = logits_np - 1e9 * (1.0 - mask_array)
                probs = softmax(masked)

                sampled_indices = []
                remaining = probs.copy()
                for _ in range(min_c):
                    if remaining.sum() <= 0:
                        break
                    p = remaining / remaining.sum()
                    idx = int(np.random.choice(len(p), p=p))
                    sampled_indices.append(idx)
                    remaining[idx] = 0.0

                choices = []
                for jax_idx in sampled_indices:
                    for cpp_idx, opt in enumerate(mock_select_dict["options"]):
                        mapped_idx = get_action_index_for_option(opt)
                        if mapped_idx == jax_idx and cpp_idx not in choices:
                            choices.append(cpp_idx)
                            break

                if len(choices) < min_c:
                    choices = list(range(min(len(opts), min_c)))
            else:
                # Random fallback
                choices = list(range(min(len(opts), min_c)))

            try:
                obs_dict = battle_select(choices)
                obs = to_dataclass(obs_dict, Observation)
            except:
                break

        # Determine result
        if obs.current and obs.current.result != -1:
            winner = obs.current.result
            # Extract reason from logs
            reason = 0
            for log in obs.logs:
                if hasattr(log, 'type') and hasattr(log, 'reason'):
                    from cg.api import LogType
                    if log.type == LogType.RESULT and log.reason is not None:
                        reason = log.reason
                        break
            return {"winner": winner, "reason": reason, "steps": steps}
        else:
            return {"winner": -1, "reason": "timeout", "steps": steps}

    # ─── Main Loop ───
    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break

        if cmd == "eval":
            deck_a, deck_b, num_games = data
            results = {"wins_p0": 0, "wins_p1": 0, "draws": 0, "steps": [], "reasons": {}}

            for g in range(num_games):
                try:
                    result = run_game(deck_a, deck_b)
                    if result["winner"] == 0:
                        results["wins_p0"] += 1
                    elif result["winner"] == 1:
                        results["wins_p1"] += 1
                    else:
                        results["draws"] += 1
                    results["steps"].append(result["steps"])
                    r = result.get("reason", 0)
                    results["reasons"][r] = results["reasons"].get(r, 0) + 1
                    battle_finish()
                except Exception as e:
                    print(f"  [Worker {worker_id}] Game error: {e}")
                    battle_finish()
                    results["draws"] += 1

            pipe.send(results)

        elif cmd == "close":
            break

    battle_finish()
    pipe.close()


class DeckEvaluator:
    """
    Evaluator pool — mengelola N worker process untuk evaluasi deck parallel.

    Usage:
        evaluator = DeckEvaluator(n_workers=4)
        results = evaluator.evaluate(deck_a, deck_b, num_games=5)
        # results = {"wins_p0": 4, "wins_p1": 1, "draws": 0, "steps": [45, 52, 38, 61, 48], "reasons": {1: 5}}
        evaluator.close()
    """

    def __init__(self, n_workers: int = 1):
        if n_workers < 1:
            n_workers = 1
        self.n_workers = n_workers
        self.pipes = []
        self.processes = []

        ctx = mp.get_context("spawn")
        for i in range(n_workers):
            parent_pipe, child_pipe = mp.Pipe()
            p = ctx.Process(target=_worker_main, args=(child_pipe, i), daemon=True)
            p.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.processes.append(p)

        print(f"[Evaluator] Started {n_workers} worker(s)")

    def evaluate(self, deck_a: list[int], deck_b: list[int], num_games: int = 5) -> dict:
        """Evaluasi deck_a (P0) vs deck_b (P1) selama num_games."""
        # Pick worker with least load (round robin)
        pipe = self.pipes[0]
        pipe.send(("eval", (deck_a, deck_b, num_games)))
        return pipe.recv()

    def evaluate_batch(self, deck: list[int], opponents: list[list[int]], num_games_per_opp: int = 3) -> dict:
        """Evaluasi deck vs multiple opponents, aggregated results."""
        total = {"wins_p0": 0, "wins_p1": 0, "draws": 0, "steps": [], "reasons": {}}
        for opp in opponents:
            result = self.evaluate(deck, opp, num_games_per_opp)
            total["wins_p0"] += result["wins_p0"]
            total["wins_p1"] += result["wins_p1"]
            total["draws"] += result["draws"]
            total["steps"].extend(result["steps"])
            for r, c in result["reasons"].items():
                total["reasons"][r] = total["reasons"].get(r, 0) + c
        return total

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except:
                pass
        for p in self.processes:
            p.join(timeout=5)
        print("[Evaluator] Closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
