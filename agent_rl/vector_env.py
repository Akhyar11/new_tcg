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
import numpy as np
import random
import os
import glob


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
