#!/usr/bin/env python3
"""
profiler.py — Profiler setiap komponen training pipeline.

Jalankan di Colab/Kaggle:
    !python profiler.py

Output:
    Profil timing per komponen, FPS breakdown, dan bottleneck terbesar.
"""
import os, sys, time, json, platform
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def clear_jax_cache():
    """Clear XLA cache between runs for clean timing."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

def wait_for_gpu_quiet():
    """Wait for GPU to settle after JIT compilation."""
    import jax.numpy as jnp
    _ = jnp.ones((1,)).block_until_ready()


def profile_component(name, n_warmup=3, n_trials=10, max_time=30.0):
    """Decorator-style: context manager for profiling."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Warmup
            for _ in range(n_warmup):
                fn(*args, **kwargs)

            timings = []
            total_start = time.perf_counter()
            for _ in range(n_trials):
                t0 = time.perf_counter()
                fn(*args, **kwargs)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)  # ms
                if time.perf_counter() - total_start > max_time:
                    break

            return {
                "name": name,
                "mean_ms": np.mean(timings),
                "median_ms": np.median(timings),
                "min_ms": np.min(timings),
                "max_ms": np.max(timings),
                "std_ms": np.std(timings),
                "trials": len(timings),
            }
        return wrapper
    return decorator


# ═══════════════════════════════════════════
# 1. GPU Inference — pure JAX
# ═══════════════════════════════════════════
def profile_gpu_inference(model, params):
    """Ukur raw JAX inference time (GPU) — warmup per batch size."""
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate, unreplicate

    apply_jit = jax.jit(model.apply)

    def bench_batch(bs, n_trials=50):
        """Benchmark satu batch size — warmup dulu baru timing."""
        dummy_seq = jnp.zeros((bs, 93, 31))
        dummy_glob = jnp.zeros((bs, 266))
        # Warmup: JIT compile untuk shape ini
        _ = apply_jit(params, dummy_seq, dummy_glob)
        wait_for_gpu_quiet()
        # Benchmark
        timings = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            logits, values = apply_jit(params, dummy_seq, dummy_glob)
            logits.block_until_ready()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)
        return timings

    t1 = bench_batch(1, 50)
    t4 = bench_batch(4, 30)
    t8 = bench_batch(8, 30)

    return {
        "batch_1_mean_ms": np.mean(t1),
        "batch_1_fps": 1000 / np.mean(t1),
        "batch_4_mean_ms": np.mean(t4),
        "batch_4_fps": 4000 / np.mean(t4),
        "batch_8_mean_ms": np.mean(t8),
        "batch_8_fps": 8000 / np.mean(t8),
        "overhead_per_env_ms": (np.mean(t8) - np.mean(t1)) / 7,
    }


# ═══════════════════════════════════════════
# 2. Feature Extraction
# ═══════════════════════════════════════════
def profile_feature_extraction():
    """Ukur waktu extract_features untuk 1 game."""
    from cg.game import battle_start, battle_finish
    from cg.api import to_dataclass, Observation
    from agent_rl.feature_extractor import extract_features

    # Load deck contoh
    deck = _load_sample_deck()

    # Start battle
    obs_dict, _ = battle_start(deck, deck)
    obs = to_dataclass(obs_dict, Observation)

    timings = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = extract_features(obs.current, obs.select, obs.current.yourIndex)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

    battle_finish()
    return {"mean_ms": np.mean(timings), "std_ms": np.std(timings)}


# ═══════════════════════════════════════════
# 3. C++ Engine — battle_select
# ═══════════════════════════════════════════
def profile_cpp_engine():
    """Ukur round-trip C++ engine untuk 1 step game."""
    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_dataclass, Observation

    deck = _load_sample_deck()
    obs_dict, _ = battle_start(deck, deck)
    obs = to_dataclass(obs_dict, Observation)

    timings = []
    for _ in range(100):
        if obs.select and obs.select.option:
            choices = [0]
            t0 = time.perf_counter()
            obs_dict = battle_select(choices)
            obs = to_dataclass(obs_dict, Observation)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)
        else:
            break

    battle_finish()
    return {"mean_ms": np.mean(timings), "std_ms": np.std(timings)}


# ═══════════════════════════════════════════
# 4. Pipe communication
# ═══════════════════════════════════════════
def _pipe_worker(pipe):
    """Module-level worker untuk pipe benchmark (harus top-level agar bisa di-pickle)."""
    import numpy as np
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'step':
                result = (
                    np.random.randn(93, 31).astype(np.float32),
                    np.random.randn(266).astype(np.float32),
                    float(np.random.randn()),
                    float(np.random.randn()),
                    {'a': 1}
                )
                pipe.send(result)
            elif cmd == 'close':
                break
    except EOFError:
        pass


def profile_pipe_communication():
    """Ukur overhead pipe send/recv untuk data volume training."""
    import multiprocessing as mp
    import numpy as np

    results = {}
    for n in [1, 2, 4, 8]:
        pipes, procs = [], []
        ctx = mp.get_context('spawn')
        for i in range(n):
            parent, child = mp.Pipe()
            p = ctx.Process(target=_pipe_worker, args=(child,))
            p.daemon = True
            p.start()
            child.close()
            pipes.append(parent)
            procs.append(p)

        logits = np.random.randn(250).astype(np.float32)
        timings = []
        for _ in range(30):
            t0 = time.perf_counter()
            for pipe in pipes:
                pipe.send(('step', logits))
            _ = [pipe.recv() for pipe in pipes]
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)

        for pipe in pipes:
            pipe.send(('close', None))
        for p in procs:
            p.join()

        results[f"{n}_envs_mean_ms"] = np.mean(timings)

    return results


# ═══════════════════════════════════════════
# 5. Full step pipeline
# ═══════════════════════════════════════════
def profile_full_step(model, params):
    """Ukur 1 step lengkap: inference → send → worker → recv → buffer."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from agent_rl.vector_env import VectorEnv

    apply_jit = jax.jit(model.apply)
    # Warmup: JIT untuk batch 8 (shape yang bakal dipakai)
    warm_seq = jnp.zeros((8, 93, 31))
    warm_glob = jnp.zeros((8, 266))
    _ = apply_jit(params, warm_seq, warm_glob)
    wait_for_gpu_quiet()

    deck_path = os.path.join(ROOT, "agent_rl", "deck_generated")
    if not os.path.exists(deck_path):
        os.makedirs(deck_path, exist_ok=True)
        _generate_dummy_decks(deck_path)

    env = VectorEnv(num_envs=8, deck_path=deck_path)
    obs = env.reset()
    seq = obs["seq_input"]
    glob = obs["glob_input"]

    breakdown = defaultdict(list)

    for step in range(50):  # kurangi jadi 50 step, buang 5 pertama
        t0 = time.perf_counter()
        logits_np = np.array(apply_jit(params, seq, glob)[0])
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        next_obs, rewards, dones, infos = env.step(logits_np)
        t3 = time.perf_counter()

        seq = next_obs["seq_input"]
        glob = next_obs["glob_input"]

        # Skip 5 first steps (cold start / JIT / worker warmup)
        if step >= 5:
            breakdown["jax_inference"].append((t1 - t0) * 1000)
            breakdown["env_step_total"].append((t3 - t2) * 1000)

    env.close()

    if not breakdown["jax_inference"]:
        return {}

    return {k: {"mean_ms": np.mean(v), "std_ms": np.std(v)}
            for k, v in breakdown.items()}


# ═══════════════════════════════════════════
# 6. PPO Update
# ═══════════════════════════════════════════
def profile_ppo_update(model, params):
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    from flax.jax_utils import replicate
    from agent_rl.ppo_update import ppo_update_step
    from agent_rl.buffer import RolloutBuffer

    num_devices = jax.device_count()
    N_STEPS, NUM_ENVS, BATCH_SIZE = 128, 8, 64

    if NUM_ENVS % num_devices != 0:
        NUM_ENVS = max((NUM_ENVS // num_devices) * num_devices, num_devices)
    if BATCH_SIZE % num_devices != 0:
        BATCH_SIZE = max((BATCH_SIZE // num_devices) * num_devices, num_devices)

    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=3e-4, eps=1e-5)
    )
    opt_state = tx.init(params)
    params_repl = replicate(params)
    opt_state_repl = replicate(opt_state)

    buffer = RolloutBuffer(n_steps=N_STEPS, num_envs=NUM_ENVS)

    # Isi buffer dengan dummy data
    seq = np.random.randn(N_STEPS, NUM_ENVS, 93, 31).astype(np.float32)
    glob = np.random.randn(N_STEPS, NUM_ENVS, 266).astype(np.float32)
    amask = np.random.binomial(1, 0.1, (N_STEPS, NUM_ENVS, 250)).astype(np.bool_)
    lp = np.random.randn(N_STEPS, NUM_ENVS).astype(np.float32)
    r = np.random.randn(N_STEPS, NUM_ENVS).astype(np.float32)
    v = np.random.randn(N_STEPS, NUM_ENVS).astype(np.float32)
    d = np.zeros((N_STEPS, NUM_ENVS), dtype=np.float32)

    for t in range(N_STEPS):
        buffer.add(seq[t], glob[t], amask[t], lp[t], r[t], v[t], d[t])

    last_v = np.random.randn(NUM_ENVS).astype(np.float32)
    buffer.compute_returns_and_advantages(last_v, d[-1], 0.99, 0.95)

    clip_ratio = 0.2
    entropy_coef = 0.05

    # Warmup
    for batch in buffer.get_batches(BATCH_SIZE):
        batch_sharded = {k: v.reshape((num_devices, BATCH_SIZE // num_devices, *v.shape[1:]))
                        for k, v in batch.items()}
        _, _, _, _ = ppo_update_step(
            params_repl, opt_state_repl, batch_sharded, model.apply, tx,
            clip_ratio, entropy_coef
        )
        break

    timings = []
    for epoch in range(4):
        for batch in buffer.get_batches(BATCH_SIZE):
            batch_sharded = {k: v.reshape((num_devices, BATCH_SIZE // num_devices, *v.shape[1:]))
                            for k, v in batch.items()}
            t0 = time.perf_counter()
            _, _, loss, _ = ppo_update_step(
                params_repl, opt_state_repl, batch_sharded, model.apply, tx,
                clip_ratio, entropy_coef
            )
            loss[0].block_until_ready()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)

    return {"mean_ms": np.mean(timings), "std_ms": np.std(timings),
            "batches": len(timings), "batch_size": BATCH_SIZE}


# ═══════════════════════════════════════════
# 7. GAE Computation
# ═══════════════════════════════════════════
def profile_gae():
    from agent_rl.buffer import RolloutBuffer
    import numpy as np

    buffer = RolloutBuffer(n_steps=128, num_envs=8)
    seq = np.random.randn(128, 8, 93, 31).astype(np.float32)
    glob = np.random.randn(128, 8, 266).astype(np.float32)
    amask = np.random.binomial(1, 0.1, (128, 8, 250)).astype(np.bool_)
    lp = np.random.randn(128, 8).astype(np.float32)
    r = np.random.randn(128, 8).astype(np.float32)
    v = np.random.randn(128, 8).astype(np.float32)
    d = np.zeros((128, 8), dtype=np.float32)

    for t in range(128):
        buffer.add(seq[t], glob[t], amask[t], lp[t], r[t], v[t], d[t])

    last_v = np.random.randn(8).astype(np.float32)

    timings = []
    for _ in range(50):
        t0 = time.perf_counter()
        buffer.compute_returns_and_advantages(last_v, d[-1], 0.99, 0.95)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

    return {"mean_ms": np.mean(timings), "std_ms": np.std(timings)}


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════
def _load_sample_deck():
    """Load deck pertama yang ditemukan."""
    deck_dir = os.path.join(ROOT, "agent_rl", "deck_generated")
    if os.path.exists(deck_dir):
        import glob
        files = sorted(glob.glob(os.path.join(deck_dir, "*.csv")))
        if files:
            with open(files[0]) as f:
                deck = [int(line.strip()) for line in f if line.strip().isdigit()]
                if len(deck) == 60:
                    return deck
    # Fallback: random valid-range IDs
    import random
    return [random.randint(1, 1267) for _ in range(60)]


def _generate_dummy_decks(path, n=10):
    """Generate dummy decks untuk profiling — isi random yang diterima engine."""
    import glob
    import random
    existing = len(glob.glob(os.path.join(path, "*.csv")))
    if existing >= n:
        return
    for i in range(n):
        deck = [random.randint(1, 1267) for _ in range(60)]
        with open(os.path.join(path, f"profiler_deck_{i}.csv"), "w") as f:
            for cid in deck:
                f.write(f"{cid}\n")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    clear_jax_cache()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 65)
    print("  PROFILER — Training Pipeline Breakdown")
    print("=" * 65)
    print(f"  Platform: {platform.platform()}")
    print(f"  Python:   {sys.version}")
    print(f"  GPU:      {jax.devices()[0].device_kind if jax.devices() else 'NONE'}")
    num_devices = jax.device_count()
    print(f"  Devices:  {num_devices}")
    print(f"  CPU:      {os.cpu_count()} logical cores")
    print("=" * 65)
    print()

    results = {}

    # ── 0. Init model ──
    print("[0/7] Loading model...")
    from agent_rl.model import PokemonAgent
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    dummy_seq = jnp.zeros((1, 93, 31))
    dummy_glob = jnp.zeros((1, 266))
    params = model.init(rng, dummy_seq, dummy_glob)
    # Load weights if exist
    cp_path = os.path.join(ROOT, "checkpoints", "model_final.msgpack")
    if os.path.exists(cp_path):
        from flax import serialization
        with open(cp_path, "rb") as f:
            params = serialization.from_bytes(params, f.read())
        print("  [OK] Loaded model_final.msgpack")
    else:
        print("  [OK] Random weights")
    print()

    # ── 1. GPU Inference ──
    print("[1/7] GPU Inference Benchmark...")
    gpu_results = profile_gpu_inference(model, params)
    results["gpu_inference"] = gpu_results
    print(f"  Batch 1: {gpu_results['batch_1_mean_ms']:.2f}ms → {gpu_results['batch_1_fps']:.0f} FPS")
    print(f"  Batch 4: {gpu_results['batch_4_mean_ms']:.2f}ms → {gpu_results['batch_4_fps']:.0f} FPS")
    print(f"  Batch 8: {gpu_results['batch_8_mean_ms']:.2f}ms → {gpu_results['batch_8_fps']:.0f} FPS")
    print()

    # ── 2. C++ Engine ──
    print("[2/7] C++ Engine (1 step)...")
    try:
        engine_results = profile_cpp_engine()
        results["cpp_engine"] = engine_results
        print(f"  {engine_results['mean_ms']:.3f}ms per step")
    except Exception as e:
        print(f"  SKIP: {e}")
    print()

    # ── 3. Feature Extraction ──
    print("[3/7] Feature Extraction (1 call)...")
    try:
        feat_results = profile_feature_extraction()
        results["feature_extraction"] = feat_results
        print(f"  {feat_results['mean_ms']:.3f}ms per call")
    except Exception as e:
        print(f"  SKIP: {e}")
    print()

    # ── 4. Pipe Communication ──
    print("[4/7] Pipe Communication (simulated)...")
    try:
        pipe_results = profile_pipe_communication()
        results["pipe"] = pipe_results
        for k, v in pipe_results.items():
            print(f"  {k}: {v:.3f}ms")
    except Exception as e:
        print(f"  SKIP: {e}")
        import traceback; traceback.print_exc()
    print()

    # ── 5. GAE ──
    print("[5/7] GAE Computation (128 steps × 8 envs)...")
    try:
        gae_results = profile_gae()
        results["gae"] = gae_results
        print(f"  {gae_results['mean_ms']:.3f}ms per compute")
    except Exception as e:
        print(f"  SKIP: {e}")
    print()

    # ── 6. PPO Update ──
    print("[6/7] PPO Update (1 gradient step)...")
    try:
        ppo_results = profile_ppo_update(model, params)
        results["ppo_update"] = ppo_results
        print(f"  {ppo_results['mean_ms']:.3f}ms per batch (batch_size={ppo_results['batch_size']})")
    except Exception as e:
        print(f"  SKIP: {e}")
        import traceback; traceback.print_exc()
    print()

    # ── 7. Full Step (JAX + Pipe + Worker) ──
    print("[7/7] Full Pipeline — 1 step end-to-end...")
    try:
        step_results = profile_full_step(model, params)
        results["full_step"] = step_results
        for k, v in step_results.items():
            print(f"  {k}: {v['mean_ms']:.3f}ms (±{v['std_ms']:.3f})")
    except Exception as e:
        print(f"  SKIP: {e}")
        import traceback; traceback.print_exc()
    print()

    # ═══════════════════════════════════════════
    # SUMMARY — FPS Breakdown
    # ═══════════════════════════════════════════
    print("=" * 65)
    print("  FPS BREAKDOWN — Per 1 Step (8 envs)")
    print("=" * 65)

    # Ambil data dari berbagai benchmark
    gi = results.get("gpu_inference", {})

    # Full-step measurement (real environment)
    fs = results.get("full_step", {})
    if fs:
        jax_ms = fs.get("jax_inference", {}).get("mean_ms", 0)
        env_ms = fs.get("env_step_total", {}).get("mean_ms", 0)
    else:
        # Fallback ke GPU batch 8
        jax_ms = gi.get("batch_8_mean_ms", 0)
        env_ms = 0

    # Data pendukung
    feat_ms = results.get("feature_extraction", {}).get("mean_ms", 0)
    pipe_ms_total = results.get("pipe", {}).get("8_envs_mean_ms", 0)
    cpp_ms = results.get("cpp_engine", {}).get("mean_ms", 0)

    # Pure GPU estimate (dari GPU benchmark bersih)
    gpu_pure = gi.get("batch_8_mean_ms", 0)

    fps_breakdown = [
        ("GPU inference (batch 8, pure)", gpu_pure),
        ("JAX inference (in pipeline, with sync)", jax_ms),
        ("Env step (send + 8 workers + recv)", env_ms),
        ("  └ C++ engine (1 call)", cpp_ms),
        ("  └ Feature extraction (1 call)", feat_ms),
        ("  └ Pipe comm (8 env × roundtrip)", pipe_ms_total),
    ]

    total = jax_ms + env_ms

    for label, ms in fps_breakdown:
        pct = (ms / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        if ms > 0:
            print(f"  {label:40s} {ms:7.2f}ms ({pct:5.1f}%)")
            if not label.startswith("  └"):
                print(f"  {'':40s} {bar}")
        else:
            print(f"  {label:40s}    N/A")

    if total > 0:
        print()
        print(f"  {40*' '} {'──────'} {'─────'}")
        print(f"  {'TOTAL per step':40s} {total:7.2f}ms")
        print(f"  {'ESTIMATED FPS (8 env)':40s} {1000/total:7.0f} FPS")
        print(f"  {'GPU theoretical peak':40s} {gi.get('batch_8_fps', 0):7.0f} FPS")

    print()
    print("=" * 65)
    print("  SISTEM INFO")
    print("=" * 65)
    print(f"  JAX version: {jax.__version__}")
    print(f"  JAX devices: {jax.devices()}")
    try:
        mem_bytes = jax.devices()[0].memory_stats() if hasattr(jax.devices()[0], 'memory_stats') else {}
        if mem_bytes:
            print(f"  GPU memory limit: {mem_bytes.get('limit', 0)/1e9:.1f}GB")
            print(f"  GPU memory in use: {mem_bytes.get('in_use', 0)/1e9:.1f}GB")
    except:
        pass

    # Save JSON
    out_path = os.path.join(ROOT, "profiler_results.json")
    # Convert numpy types for JSON
    def convert(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            return {k: convert(v) for k, v in v.items()}
        return v
    results_clean = convert(results)
    with open(out_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\n[OK] Results saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
