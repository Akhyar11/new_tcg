"""
VectorEnv — Paralel environment manager untuk PPO training.

v3 — Non-Symmetric Opponents
=============================
P0 dan P1 selalu memulai dengan deck BERBEDA.
P1 menggunakan deck dari deck_generated/ (1000 deck diverse).
P0 menggunakan deck dari deck_generated/ JUGA tapi berbeda dari P1.

Ini memecah simetri self-play: terminal gradient tidak saling membatalkan
karena starting deck berbeda → win rate tidak selalu 50%.

Alur deck assignment per game:
  1. P0 deck = random sample dari loaded_decks
  2. P1 deck = random sample dari loaded_decks (diFFERENT from P0)
  3. Kedua player menggunakan model YANG SAMA untuk bermain
  4. Karena deck berbeda, satu sisi punya keunggulan → gradient jelas

Untuk Kaggle: cukup 1 folder deck_generated/ dengan 1000 deck random.
"""
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import random
import os
import glob
import time


def load_deck(filepath):
    """Load deck dari CSV (satu card ID per baris)."""
    deck = []
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                deck.append(int(line))
    if len(deck) != 60:
        return None
    return deck


def softmax(x):
    """Numerically stable softmax."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / (exp_x.sum() + 1e-10)


def worker(remote, parent_remote, worker_id, deck_path):
    """
    Worker independen di sub-process.
    Menangani eksekusi aksi, sampling, dan reward calculation.
    """
    parent_remote.close()

    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_dataclass, Observation, LogType, OptionType
    from agent_rl.feature_extractor import extract_features
    from agent_rl.reward import detect_events, calculate_step_reward, reset_trackers
    from agent_rl.action_mapping import decode_action, get_action_index_for_option

    # Pre-load decks
    deck_files = []
    deck_paths_checked = []

    # Cari deck di beberapa kemungkinan lokasi
    possible_paths = [
        deck_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_generated"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_rl", "deck_generated"),
    ]
    for p in possible_paths:
        if p not in deck_paths_checked:
            deck_paths_checked.append(p)
            if os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, "*.csv")))
                deck_files.extend(files)
            elif os.path.isfile(p):
                deck_files.append(p)

    # Deduplicate
    deck_files = list(dict.fromkeys(deck_files))

    loaded_decks = []
    for f in deck_files:
        d = load_deck(f)
        if d is not None:
            loaded_decks.append(d)

    if len(loaded_decks) < 2:
        print(f"[Worker {worker_id}] WARNING: hanya {len(loaded_decks)} deck valid. "
              f"Menggunakan deck dummy fallback.")
        loaded_decks = [[1]*56 + [210]*4]

    print(f"[Worker {worker_id}] Loaded {len(loaded_decks)} decks from {len(deck_files)} files")

    obs = None
    old_state = None  # Untuk deteksi event antar-step
    game_step_counter = 0  # ⭐ Counter langkah dalam satu game
    MAX_GAME_STEPS = 300   # ⭐ Safety limit — normal game ~100-150 steps

    def get_end_reason(obs_data) -> int:
        if obs_data is None or not obs_data.logs:
            return 0
        for log in obs_data.logs:
            if log.type == LogType.RESULT:
                return log.reason if log.reason is not None else 0
        return 0

    empty_features = {
        "seq_input": np.zeros((93, 31), dtype=np.float32),
        "glob_input": np.zeros(266, dtype=np.float32)
    }

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                logits = data  # (250,) numpy array

                mock_select_dict = {"options": []}
                min_c = 1
                if obs and obs.select and obs.select.option:
                    mock_select_dict = {
                        "options": [
                            {"type": OptionType(o.type).name, "index": o.index}
                            for o in obs.select.option
                        ]
                    }
                    min_c = obs.select.minCount

                options = mock_select_dict["options"]

                # 1. Build action mask
                legal_mask = np.zeros(250, dtype=np.float32)
                for opt in options:
                    idx = get_action_index_for_option(opt)
                    if 0 <= idx < 250:
                        legal_mask[idx] = 1.0

                # 2. Mask logits
                masked_logits = logits - 1e9 * (1.0 - legal_mask)

                # 3. Softmax → probs
                probs = softmax(masked_logits)

                # 4. Sample min_c actions tanpa replacement
                if probs.sum() > 0:
                    remaining = probs.copy()
                    sampled_jax_indices = []
                    for _ in range(min_c):
                        if remaining.sum() <= 0:
                            break
                        p = remaining / remaining.sum()
                        idx = np.random.choice(len(p), p=p)
                        sampled_jax_indices.append(int(idx))
                        remaining[idx] = 0.0
                else:
                    sampled_jax_indices = [160]  # ACTION_END

                # 5. Map ke C++ indices
                choices = []
                for jax_idx in sampled_jax_indices:
                    for cpp_idx, opt in enumerate(options):
                        mapped_idx = get_action_index_for_option(opt)
                        if mapped_idx == jax_idx and cpp_idx not in choices:
                            choices.append(cpp_idx)
                            break

                if len(choices) < min_c:
                    for cpp_idx in range(len(options)):
                        if cpp_idx not in choices:
                            choices.append(cpp_idx)
                        if len(choices) >= min_c:
                            break

                # 6. Build actions_mask
                actions_mask = np.zeros(250, dtype=np.bool_)
                for c in choices:
                    if c < len(options):
                        idx = get_action_index_for_option(options[c])
                        if 0 <= idx < 250:
                            actions_mask[idx] = True

                if not np.any(actions_mask):
                    actions_mask[160] = True
                    choices = [0]

                # Simpan player index SEBELUM execute
                prev_player = obs.current.yourIndex if obs.current else 0

                # ⭐ Increment step counter untuk safety limit
                game_step_counter += 1

                # ⭐ Force-reset jika game terlalu panjang (engine stuck / R9 loop)
                if game_step_counter >= MAX_GAME_STEPS:
                    # Force end: treat sebagai draw, reward 0
                    battle_finish()
                    done = True
                    reward = 0.0
                    end_reason = 9  # Timeout / stuck
                    events = {}
                    actions_mask = np.zeros(250, dtype=np.bool_)
                    actions_mask[160] = True  # ACTION_END

                    # Auto-reset untuk game berikutnya
                    reset_trackers()
                    try:
                        if len(loaded_decks) >= 2:
                            idx0 = random.randint(0, len(loaded_decks) - 1)
                            idx1 = random.randint(0, len(loaded_decks) - 1)
                            while idx1 == idx0 and len(loaded_decks) > 1:
                                idx1 = random.randint(0, len(loaded_decks) - 1)
                            deck_list = [loaded_decks[idx0], loaded_decks[idx1]]
                        else:
                            deck_list = [loaded_decks[0], loaded_decks[0]]

                        obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                        obs = to_dataclass(obs_dict, Observation)
                        old_state = obs.current
                    except Exception as e:
                        print(f"[Worker {worker_id}] Force-reset error: {e}")
                        obs = Observation(current=None, select=None, logs=[])
                        old_state = None

                    game_step_counter = 0

                    if obs.current and obs.select and obs.current.result == -1:
                        features = extract_features(obs.current, obs.select, obs.current.yourIndex)
                    else:
                        features = empty_features

                    remote.send((features, reward, done, {
                        "actions_mask": actions_mask,
                        "glob_mask": np.zeros(250, dtype=np.float32),
                        "active_player": prev_player,
                        "result": -1,
                        "end_reason": end_reason
                    }))
                    continue  # ⭐ Skip sisa step handler — sudah kirim response

                # Execute di C++ engine
                try:
                    obs_dict = battle_select(choices)
                    obs = to_dataclass(obs_dict, Observation)
                except Exception as e:
                    try:
                        opt_count = len(obs.select.option) if obs.select and obs.select.option else 0
                        min_c = obs.select.minCount if obs.select else 0
                        fallback = list(range(min(opt_count, min_c))) if min_c > 0 else []
                        obs_dict = battle_select(fallback)
                        obs = to_dataclass(obs_dict, Observation)
                    except:
                        obs = Observation(current=None, select=None, logs=[])

                # Hitung reward untuk prev_player
                if obs.current:
                    end_reason = get_end_reason(obs)
                    events = detect_events(old_state, obs.current, prev_player, obs.logs)
                    reward = calculate_step_reward(obs.current, prev_player, events, end_reason)
                    old_state = obs.current
                    done = (obs.current.result != -1)
                    active_p = prev_player
                else:
                    active_p = prev_player
                    reward = -2.0 if prev_player == 0 else -2.0
                    done = True
                    events = {}

                info = {
                    "actions_mask": actions_mask,
                    "glob_mask": legal_mask.astype(np.float32),
                    "active_player": active_p,
                    "result": obs.current.result if obs.current else -1,
                    "end_reason": get_end_reason(obs) if done else 0
                }

                # Auto-reset jika game selesai
                if done:
                    game_step_counter = 0
                    reset_trackers()
                    battle_finish()
                    try:
                        # P0 dan P1 selalu dapat deck BERBEDA
                        if len(loaded_decks) >= 2:
                            idx0 = random.randint(0, len(loaded_decks) - 1)
                            idx1 = random.randint(0, len(loaded_decks) - 1)
                            while idx1 == idx0 and len(loaded_decks) > 1:
                                idx1 = random.randint(0, len(loaded_decks) - 1)
                            deck_list = [loaded_decks[idx0], loaded_decks[idx1]]
                        else:
                            deck_list = [loaded_decks[0], loaded_decks[0]]

                        obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                        obs = to_dataclass(obs_dict, Observation)
                        old_state = obs.current

                        if obs.current and obs.select and obs.current.result == -1:
                            features = extract_features(obs.current, obs.select, obs.current.yourIndex)
                        else:
                            features = empty_features
                    except Exception as e:
                        print(f"[Worker {worker_id}] Auto-reset error: {e}")
                        obs = Observation(current=None, select=None, logs=[])
                        features = empty_features
                        old_state = None
                else:
                    if obs.current and obs.select and obs.current.result == -1:
                        features = extract_features(obs.current, obs.select, obs.current.yourIndex)
                    else:
                        features = empty_features

                remote.send((features, reward, done, info))

            elif cmd == 'reset':
                game_step_counter = 0
                reset_trackers()
                battle_finish()
                try:
                    # P0 dan P1 deck berbeda
                    if len(loaded_decks) >= 2:
                        idx0 = random.randint(0, len(loaded_decks) - 1)
                        idx1 = random.randint(0, len(loaded_decks) - 1)
                        while idx1 == idx0 and len(loaded_decks) > 1:
                            idx1 = random.randint(0, len(loaded_decks) - 1)
                        deck_list = [loaded_decks[idx0], loaded_decks[idx1]]
                    else:
                        deck_list = [loaded_decks[0], loaded_decks[0]]

                    obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                    obs = to_dataclass(obs_dict, Observation)
                    old_state = obs.current

                    if obs.current and obs.select and obs.current.result == -1:
                        features = extract_features(obs.current, obs.select, obs.current.yourIndex)
                    else:
                        features = empty_features
                except Exception as e:
                    print(f"[Worker {worker_id}] Reset error: {e}")
                    obs = Observation(current=None, select=None, logs=[])
                    features = empty_features
                    old_state = None

                remote.send(features)

            elif cmd == 'close':
                battle_finish()
                remote.close()
                break

        except EOFError:
            break
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            remote.send((empty_features, -2.0, True, {
                "actions_mask": np.zeros(250, dtype=np.bool_),
                "glob_mask": np.zeros(250, dtype=np.float32),
                "active_player": 0,
                "result": -1,
                "end_reason": 0
            }))


class VectorEnv:
    """
    Manajer Lingkungan Paralel.
    P0 dan P1 selalu mendapat deck BERBEDA → self-play gradient tidak saling cancel.
    """
    def __init__(self, num_envs, deck_path="agent_rl/deck_generated"):
        self.num_envs = num_envs
        # Auto-resolve deck_path
        if not os.path.exists(deck_path):
            alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_generated")
            if os.path.exists(alt):
                deck_path = alt
            else:
                alt2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "agent_rl", "deck_generated")
                if os.path.exists(alt2):
                    deck_path = alt2

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        ctx = mp.get_context('spawn')

        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=worker, args=(work_remote, remote, i, deck_path))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        results = [remote.recv() for remote in self.remotes]
        seq_inputs = np.stack([res["seq_input"] for res in results])
        glob_inputs = np.stack([res["glob_input"] for res in results])
        return {"seq_input": seq_inputs, "glob_input": glob_inputs}

    def step_async(self, logits_batch):
        for remote, logits in zip(self.remotes, logits_batch):
            remote.send(('step', logits))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]

        seq_inputs = np.stack([res[0]["seq_input"] for res in results])
        glob_inputs = np.stack([res[0]["glob_input"] for res in results])
        batch_features = {"seq_input": seq_inputs, "glob_input": glob_inputs}

        rewards = np.array([res[1] for res in results], dtype=np.float32)
        dones = np.array([res[2] for res in results], dtype=np.bool_)
        infos = [res[3] for res in results]

        return batch_features, rewards, dones, infos

    def step(self, logits_batch):
        self.step_async(logits_batch)
        return self.step_wait()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()


# ═══════════════════════════════════════════════════════════
# ShmVectorEnv — Shared Memory VectorEnv (zero-copy data)
# ═══════════════════════════════════════════════════════════
# Menggunakan shared memory untuk semua data transfer.
# Pipe hanya untuk sync signal (1 int per step per worker).
# Eliminasi semua pickle serialization overhead.
#
# Shared memory layout per worker (14864 bytes):
#   [0:1000]     logits       (250 float32)
#   [1000:12532] seq_input    (93×31 float32)
#   [12532:13596] glob_input  (266 float32)
#   [13596:13600] reward      (1 float32)
#   [13600:13601] done        (1 int8)
#   [13601:13851] actions_mask (250 bool)
#   [13851:14851] glob_mask   (250 float32)
#   [14851:14855] result      (1 int32)
#   [14855:14859] end_reason  (1 int32)
#   ──────────────────────────────────
#   Total: 14859 → aligned 14864
# ═══════════════════════════════════════════════════════════

_SHM_STRIDE = 14864
_SHM_LOGITS_END = 1000
_SHM_SEQ_END = 12532
_SHM_GLOB_END = 13596


def _worker_shm(remote, parent_remote, worker_id, deck_path, shm_name):
    """Worker process dengan shared memory — zero-copy data transfer."""
    parent_remote.close()

    # Attach ke shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    off = worker_id * _SHM_STRIDE

    # Buat numpy views langsung di shared memory
    logits_view = np.ndarray((250,), dtype=np.float32, buffer=shm.buf[off:off+1000])
    seq_view = np.ndarray((93, 31), dtype=np.float32, buffer=shm.buf[off+1000:off+_SHM_SEQ_END])
    glob_view = np.ndarray((266,), dtype=np.float32, buffer=shm.buf[off+_SHM_SEQ_END:off+_SHM_GLOB_END])
    reward_view = np.ndarray((1,), dtype=np.float32, buffer=shm.buf[off+13596:off+13600])
    done_view = np.ndarray((1,), dtype=np.int8, buffer=shm.buf[off+13600:off+13601])
    amask_view = np.ndarray((250,), dtype=np.bool_, buffer=shm.buf[off+13601:off+13851])
    gmask_view = np.ndarray((250,), dtype=np.float32, buffer=shm.buf[off+13851:off+14851])
    result_view = np.ndarray((1,), dtype=np.int32, buffer=shm.buf[off+14851:off+14855])
    ereason_view = np.ndarray((1,), dtype=np.int32, buffer=shm.buf[off+14855:off+14859])

    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_dataclass, Observation, LogType, OptionType
    from agent_rl.feature_extractor import extract_features
    from agent_rl.reward import detect_events, calculate_step_reward, reset_trackers
    from agent_rl.action_mapping import get_action_index_for_option

    # Load decks
    deck_files = []
    for p in [
        deck_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_generated"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_rl", "deck_generated"),
    ]:
        if os.path.isdir(p):
            deck_files.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
        elif os.path.isfile(p):
            deck_files.append(p)
    deck_files = list(dict.fromkeys(deck_files))
    loaded_decks = []
    for f in deck_files:
        d = load_deck(f)
        if d is not None:
            loaded_decks.append(d)
    if len(loaded_decks) < 2:
        loaded_decks = [[1]*56 + [210]*4]

    obs = None
    old_state = None
    game_step_counter = 0
    MAX_GAME_STEPS = 300
    empty_seq = np.zeros((93, 31), dtype=np.float32)
    empty_glob = np.zeros(266, dtype=np.float32)

    def get_end_reason(obs_data):
        if obs_data is None or not obs_data.logs:
            return 0
        for log in obs_data.logs:
            if log.type == LogType.RESULT:
                return log.reason if log.reason is not None else 0
        return 0

    while True:
        try:
            cmd = remote.recv()
            if isinstance(cmd, (list, tuple)):
                if len(cmd) > 0:
                    cmd = cmd[0]
                else:
                    cmd = 'step'

            if cmd == 'step':
                # ── 1. Baca logits dari shared memory ──
                logits = logits_view.copy()

                # ── 2. Action sampling ──
                mock_select_dict = {"options": []}
                min_c = 1
                if obs and obs.select and obs.select.option:
                    mock_select_dict = {
                        "options": [
                            {"type": OptionType(o.type).name, "index": o.index}
                            for o in obs.select.option
                        ]
                    }
                    min_c = obs.select.minCount

                options = mock_select_dict["options"]
                legal_mask = np.zeros(250, dtype=np.float32)
                for opt in options:
                    idx = get_action_index_for_option(opt)
                    if 0 <= idx < 250:
                        legal_mask[idx] = 1.0

                masked = logits - 1e9 * (1.0 - legal_mask)
                probs = softmax(masked)
                if probs.sum() > 0:
                    remaining = probs.copy()
                    sampled = []
                    for _ in range(min_c):
                        if remaining.sum() <= 0:
                            break
                        p = remaining / remaining.sum()
                        sampled.append(int(np.random.choice(len(p), p=p)))
                        remaining[sampled[-1]] = 0.0
                else:
                    sampled = [160]

                choices = []
                for jdx in sampled:
                    for cpp_idx, opt in enumerate(options):
                        mapped = get_action_index_for_option(opt)
                        if mapped == jdx and cpp_idx not in choices:
                            choices.append(cpp_idx)
                            break
                while len(choices) < min_c:
                    for cpp_idx in range(len(options)):
                        if cpp_idx not in choices:
                            choices.append(cpp_idx)
                            break
                    else:
                        choices.append(0)

                actions_mask = np.zeros(250, dtype=np.bool_)
                for c in choices:
                    if c < len(options):
                        idx = get_action_index_for_option(options[c])
                        if 0 <= idx < 250:
                            actions_mask[idx] = True
                if not np.any(actions_mask):
                    actions_mask[160] = True

                prev_player = obs.current.yourIndex if obs.current else 0
                game_step_counter += 1

                # ── 3. Force-reset jika game terlalu panjang ──
                if game_step_counter >= MAX_GAME_STEPS:
                    battle_finish()
                    seq_view[:] = empty_seq
                    glob_view[:] = empty_glob
                    reward_view[0] = 0.0
                    done_view[0] = 1
                    amask_view[:] = False
                    amask_view[160] = True
                    gmask_view[:] = 0.0
                    result_view[0] = -1
                    ereason_view[0] = 9
                    reset_trackers()
                    try:
                        idx0 = random.randint(0, len(loaded_decks)-1)
                        idx1 = random.randint(0, len(loaded_decks)-1)
                        while idx1 == idx0 and len(loaded_decks) > 1:
                            idx1 = random.randint(0, len(loaded_decks)-1)
                        obs_dict, _ = battle_start(loaded_decks[idx0], loaded_decks[idx1])
                        obs = to_dataclass(obs_dict, Observation)
                        old_state = obs.current
                    except Exception:
                        obs = Observation(current=None, select=None, logs=[])
                        old_state = None
                    game_step_counter = 0
                    remote.send(0)
                    continue

                # ── 4. Execute C++ engine ──
                try:
                    obs_dict = battle_select(choices)
                    obs = to_dataclass(obs_dict, Observation)
                except Exception:
                    try:
                        opt_count = len(obs.select.option) if obs.select and obs.select.option else 0
                        mc = obs.select.minCount if obs.select else 0
                        fallback = list(range(min(opt_count, mc))) if mc > 0 else []
                        obs_dict = battle_select(fallback)
                        obs = to_dataclass(obs_dict, Observation)
                    except Exception:
                        obs = Observation(current=None, select=None, logs=[])

                # ── 5. Hitung reward + extract features ──
                if obs.current:
                    end_reason = get_end_reason(obs)
                    events = detect_events(old_state, obs.current, prev_player, obs.logs)
                    reward = calculate_step_reward(obs.current, prev_player, events, end_reason)
                    old_state = obs.current
                    done = (obs.current.result != -1)

                    feats = extract_features(obs.current, obs.select, obs.current.yourIndex)
                    seq_view[:] = feats["seq_input"]
                    glob_view[:] = feats["glob_input"]
                    reward_view[0] = reward
                    done_view[0] = 1 if done else 0
                    amask_view[:] = actions_mask
                    gmask_view[:] = legal_mask
                    result_view[0] = obs.current.result or 0
                    ereason_view[0] = end_reason if done else 0
                else:
                    seq_view[:] = empty_seq
                    glob_view[:] = empty_glob
                    reward_view[0] = -2.0
                    done_view[0] = 1
                    amask_view[:] = False
                    amask_view[160] = True
                    gmask_view[:] = 0.0
                    result_view[0] = -1
                    ereason_view[0] = 0
                    done = True

                # ── 6. Auto-reset ──
                if done:
                    game_step_counter = 0
                    reset_trackers()
                    battle_finish()
                    try:
                        idx0 = random.randint(0, len(loaded_decks)-1)
                        idx1 = random.randint(0, len(loaded_decks)-1)
                        while idx1 == idx0 and len(loaded_decks) > 1:
                            idx1 = random.randint(0, len(loaded_decks)-1)
                        obs_dict, _ = battle_start(loaded_decks[idx0], loaded_decks[idx1])
                        obs = to_dataclass(obs_dict, Observation)
                        old_state = obs.current
                    except Exception:
                        obs = Observation(current=None, select=None, logs=[])
                        old_state = None

                remote.send(0)

            elif cmd == 'reset':
                game_step_counter = 0
                reset_trackers()
                battle_finish()
                try:
                    idx0 = random.randint(0, len(loaded_decks)-1)
                    idx1 = random.randint(0, len(loaded_decks)-1)
                    while idx1 == idx0 and len(loaded_decks) > 1:
                        idx1 = random.randint(0, len(loaded_decks)-1)
                    obs_dict, _ = battle_start(loaded_decks[idx0], loaded_decks[idx1])
                    obs = to_dataclass(obs_dict, Observation)
                    old_state = obs.current
                    if obs.current and obs.select and obs.current.result == -1:
                        feats = extract_features(obs.current, obs.select, obs.current.yourIndex)
                        seq_view[:] = feats["seq_input"]
                        glob_view[:] = feats["glob_input"]
                    else:
                        seq_view[:] = empty_seq
                        glob_view[:] = empty_glob
                except Exception:
                    seq_view[:] = empty_seq
                    glob_view[:] = empty_glob
                    obs = Observation(current=None, select=None, logs=[])
                    old_state = None
                remote.send(0)

            elif cmd == 'close':
                battle_finish()
                shm.close()
                remote.close()
                break

        except EOFError:
            break
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            remote.send(-1)

    shm.close()


class ShmVectorEnv:
    """
    VectorEnv dengan shared memory — zero-copy worker communication.

    API IDENTIK dengan VectorEnv (drop-in replacement).
    Pipe hanya untuk command sync (1 int per step), bukan untuk data.
    Data (logits, features, rewards) via shared memory → no pickle overhead.

    Gunakan:
        env = ShmVectorEnv(num_envs=8, deck_path="agent_rl/deck_generated")
        obs = env.reset()
        for step in range(N_STEPS):
            obs, rewards, dones, infos = env.step(logits)
    """

    def __init__(self, num_envs, deck_path="agent_rl/deck_generated"):
        self.num_envs = num_envs

        # Auto-resolve deck_path
        if not os.path.exists(deck_path):
            for alt in [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_generated"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "agent_rl", "deck_generated"),
            ]:
                if os.path.exists(alt):
                    deck_path = alt
                    break

        # ═══ Create shared memory ═══
        total_bytes = num_envs * _SHM_STRIDE
        self.shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        self.shm_name = self.shm.name

        # Buat numpy views di main process untuk quick write/read
        self._seq_arr = np.zeros((num_envs, 93, 31), dtype=np.float32)
        self._glob_arr = np.zeros((num_envs, 266), dtype=np.float32)
        self._reward_arr = np.zeros(num_envs, dtype=np.float32)
        self._done_arr = np.zeros(num_envs, dtype=np.bool_)
        self._amask_arr = np.zeros((num_envs, 250), dtype=np.bool_)
        self._gmask_arr = np.zeros((num_envs, 250), dtype=np.float32)
        self._result_arr = np.zeros(num_envs, dtype=np.int32)
        self._ereason_arr = np.zeros(num_envs, dtype=np.int32)

        # Untuk step_async: pre-allocate per-worker logits views
        self._logits_views = []
        for i in range(num_envs):
            off = i * _SHM_STRIDE
            v = np.ndarray((250,), dtype=np.float32,
                           buffer=self.shm.buf[off:off+1000])
            self._logits_views.append(v)

        # ═══ Start worker processes ═══
        self.remotes = []  # pipes for sync only
        self.processes = []
        ctx = mp.get_context('spawn')

        for i in range(num_envs):
            parent_pipe, child_pipe = mp.Pipe(duplex=True)
            p = ctx.Process(target=_worker_shm,
                           args=(child_pipe, parent_pipe, i, deck_path, self.shm_name))
            p.daemon = True
            p.start()
            child_pipe.close()
            self.remotes.append(parent_pipe)
            self.processes.append(p)

    def reset(self):
        """Reset semua env. Features langsung dari shared memory."""
        for r in self.remotes:
            r.send('reset')

        for r in self.remotes:
            r.recv()  # sync

        # Baca dari shared memory
        for i in range(self.num_envs):
            off = i * _SHM_STRIDE
            self._seq_arr[i] = np.frombuffer(
                self.shm.buf[off+1000:off+_SHM_SEQ_END], dtype=np.float32).reshape(93, 31)
            self._glob_arr[i] = np.frombuffer(
                self.shm.buf[off+_SHM_SEQ_END:off+_SHM_GLOB_END], dtype=np.float32)

        return {"seq_input": self._seq_arr.copy(), "glob_input": self._glob_arr.copy()}

    def step_async(self, logits_batch):
        """Tulis logits ke shared memory, kirim command ke workers."""
        for i, logits in enumerate(logits_batch):
            self._logits_views[i][:] = logits
        for r in self.remotes:
            r.send('step')

    def step_wait(self):
        """Baca hasil dari shared memory setelah workers selesai."""
        for r in self.remotes:
            r.recv()  # sync

        # Baca semua hasil dari shared memory — zero copy read
        for i in range(self.num_envs):
            off = i * _SHM_STRIDE
            # seq (93x31 float32)
            self._seq_arr[i] = np.frombuffer(
                self.shm.buf[off+1000:off+_SHM_SEQ_END], dtype=np.float32).reshape(93, 31)
            # glob (266 float32)
            self._glob_arr[i] = np.frombuffer(
                self.shm.buf[off+_SHM_SEQ_END:off+_SHM_GLOB_END], dtype=np.float32)
            # reward (1 float32)
            self._reward_arr[i] = np.frombuffer(
                self.shm.buf[off+13596:off+13600], dtype=np.float32)[0]
            # done (1 int8 → bool)
            self._done_arr[i] = bool(
                np.frombuffer(self.shm.buf[off+13600:off+13601], dtype=np.int8)[0])
            # actions_mask (250 bool)
            self._amask_arr[i] = np.frombuffer(
                self.shm.buf[off+13601:off+13851], dtype=np.bool_)
            # glob_mask (250 float32)
            self._gmask_arr[i] = np.frombuffer(
                self.shm.buf[off+13851:off+14851], dtype=np.float32)
            # result (1 int32)
            self._result_arr[i] = np.frombuffer(
                self.shm.buf[off+14851:off+14855], dtype=np.int32)[0]
            # end_reason (1 int32)
            self._ereason_arr[i] = np.frombuffer(
                self.shm.buf[off+14855:off+14859], dtype=np.int32)[0]

        batch_features = {"seq_input": self._seq_arr.copy(),
                         "glob_input": self._glob_arr.copy()}

        infos = [
            {
                "actions_mask": self._amask_arr[i],
                "glob_mask": self._gmask_arr[i],
                "active_player": 0,
                "result": self._result_arr[i],
                "end_reason": self._ereason_arr[i],
            }
            for i in range(self.num_envs)
        ]

        return batch_features, self._reward_arr.copy(), self._done_arr.copy(), infos

    def step(self, logits_batch):
        self.step_async(logits_batch)
        return self.step_wait()

    def close(self):
        for r in self.remotes:
            r.send('close')
        for p in self.processes:
            p.join(timeout=5)
        if hasattr(self, 'shm') and self.shm:
            self.shm.close()
            try:
                self.shm.unlink()
            except Exception:
                pass
