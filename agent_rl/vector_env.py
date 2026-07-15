"""
VectorEnv — Paralel environment manager untuk PPO training.

v4 — Shared Memory Optimization
=============================
Memanfaatkan multiprocessing.shared_memory untuk komunikasi data
tanpa serialisasi/pickling overhead. Kecepatan IPC meningkat drastis.
"""
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
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


def worker(remote, parent_remote, worker_id, new_deck_path, gen_deck_path, num_envs, shm_names):
    """
    Worker independen di sub-process menggunakan Shared Memory.
    Menangani eksekusi aksi, sampling, dan reward calculation.
    """
    parent_remote.close()

    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_dataclass, Observation, LogType, OptionType
    from agent_rl.feature_extractor import extract_features
    from agent_rl.reward import detect_events, calculate_step_reward, reset_trackers
    from agent_rl.action_mapping import decode_action, get_action_index_for_option

    # Attach to Shared Memories
    shms = {k: SharedMemory(name=v) for k, v in shm_names.items()}
    
    seq_input_buf = np.ndarray((num_envs, 113, 31), dtype=np.float32, buffer=shms['seq_input'].buf)[worker_id]
    glob_input_buf = np.ndarray((num_envs, 266), dtype=np.float32, buffer=shms['glob_input'].buf)[worker_id]
    logits_buf = np.ndarray((num_envs, 250), dtype=np.float32, buffer=shms['logits'].buf)[worker_id]
    
    rewards_buf = np.ndarray((num_envs,), dtype=np.float32, buffer=shms['rewards'].buf)
    dones_buf = np.ndarray((num_envs,), dtype=np.bool_, buffer=shms['dones'].buf)
    
    actions_mask_buf = np.ndarray((num_envs, 250), dtype=np.bool_, buffer=shms['actions_mask'].buf)[worker_id]
    glob_mask_buf = np.ndarray((num_envs, 250), dtype=np.float32, buffer=shms['glob_mask'].buf)[worker_id]
    
    active_player_buf = np.ndarray((num_envs,), dtype=np.int32, buffer=shms['active_player'].buf)
    turn_changed_buf = np.ndarray((num_envs,), dtype=np.bool_, buffer=shms['turn_changed'].buf)
    result_buf = np.ndarray((num_envs,), dtype=np.int32, buffer=shms['result'].buf)
    end_reason_buf = np.ndarray((num_envs,), dtype=np.int32, buffer=shms['end_reason'].buf)

    # Pre-load decks
    def get_loaded_decks(target_path):
        deck_files = []
        deck_paths_checked = []

        possible_paths = [
            target_path,
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

        deck_files = list(dict.fromkeys(deck_files))
        loaded = []
        for f in deck_files:
            d = load_deck(f)
            if d is not None:
                loaded.append(d)

        if len(loaded) < 1:
            print(f"[Worker {worker_id}] WARNING: tidak ada deck valid di {target_path}. Menggunakan fallback.")
            loaded = [[1]*56 + [210]*4]
        return loaded

    loaded_new_decks = get_loaded_decks(new_deck_path)
    loaded_gen_decks = get_loaded_decks(gen_deck_path)

    print(f"[Worker {worker_id}] Loaded {len(loaded_new_decks)} New Decks (70%) and {len(loaded_gen_decks)} Gen Decks (30%)")

    def sample_deck():
        """Pilih deck dengan peluang 70% New Deck, 30% Generated Deck"""
        if random.random() < 0.70 and len(loaded_new_decks) > 0:
            return random.choice(loaded_new_decks)
        elif len(loaded_gen_decks) > 0:
            return random.choice(loaded_gen_decks)
        elif len(loaded_new_decks) > 0:
            return random.choice(loaded_new_decks)
        else:
            return [1]*56 + [210]*4

    obs = None
    old_state = None
    game_step_counter = 0
    MAX_GAME_STEPS = 300
    opp_known_hand = []

    def get_end_reason(obs_data) -> int:
        if obs_data is None or not obs_data.logs:
            return 0
        for log in obs_data.logs:
            if log.type == LogType.RESULT:
                return log.reason if log.reason is not None else 0
        return 0

    try:
        while True:
            cmd = remote.recv()

            if cmd == 'step':
                if not obs or not obs.current:
                    # Lingkungan dalam kondisi crash/broken dari iterasi sebelumnya. Langsung pancing auto-reset.
                    rewards_buf[worker_id] = -2.0
                    dones_buf[worker_id] = True
                    actions_mask_buf.fill(False)
                    glob_mask_buf.fill(0)
                    active_player_buf[worker_id] = 0
                    result_buf[worker_id] = -1
                    end_reason_buf[worker_id] = 0
                    turn_changed_buf[worker_id] = False
                    
                    # Manual auto-reset (simulating if done: loop)
                    game_step_counter = 0
                    reset_trackers()
                    opp_known_hand.clear()
                    battle_finish()
                    
                    success = False
                    for _ in range(10):
                        try:
                            deck_list = [sample_deck(), sample_deck()]

                            obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                            obs = to_dataclass(obs_dict, Observation)
                            old_state = obs.current

                            if obs.current and obs.select and obs.current.result == -1:
                                features = extract_features(obs.current, obs.select, obs.current.yourIndex, opp_known_hand)
                                np.copyto(seq_input_buf, features['seq_input'])
                                np.copyto(glob_input_buf, features['glob_input'])
                                success = True
                                break
                        except Exception as e:
                            battle_finish()
                            
                    if not success:
                        deck_list = [[1]*56 + [210]*4, [1]*56 + [210]*4]
                        try:
                            obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                            obs = to_dataclass(obs_dict, Observation)
                            old_state = obs.current
                            if obs.current and obs.select and obs.current.result == -1:
                                features = extract_features(obs.current, obs.select, obs.current.yourIndex, opp_known_hand)
                                np.copyto(seq_input_buf, features['seq_input'])
                                np.copyto(glob_input_buf, features['glob_input'])
                        except:
                            obs = Observation(current=None, select=None, logs=[])
                            old_state = None
                            seq_input_buf.fill(0)
                            glob_input_buf.fill(0)

                    remote.send('done')
                    continue

                logits = logits_buf.copy()

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

                masked_logits = logits - 1e9 * (1.0 - legal_mask)
                probs = softmax(masked_logits)

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
                    sampled_jax_indices = [160]

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

                actions_mask = np.zeros(250, dtype=np.bool_)
                for c in choices:
                    if c < len(options):
                        idx = get_action_index_for_option(options[c])
                        if 0 <= idx < 250:
                            actions_mask[idx] = True

                if not np.any(actions_mask):
                    actions_mask[160] = True
                    choices = [0]

                prev_player = obs.current.yourIndex if obs.current else 0
                game_step_counter += 1

                if game_step_counter >= MAX_GAME_STEPS:
                    battle_finish()
                    done = True
                    reward = 0.0
                    end_reason = 9
                    
                    actions_mask = np.zeros(250, dtype=np.bool_)
                    actions_mask[160] = True

                    reset_trackers()
                    opp_known_hand.clear()
                    try:
                        deck_list = [sample_deck(), sample_deck()]

                        obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                        obs = to_dataclass(obs_dict, Observation)
                        old_state = obs.current
                    except Exception as e:
                        print(f"[Worker {worker_id}] Force-reset error: {e}")
                        obs = Observation(current=None, select=None, logs=[])
                        old_state = None

                    game_step_counter = 0

                    if obs.current and obs.select and obs.current.result == -1:
                        features = extract_features(obs.current, obs.select, obs.current.yourIndex, opp_known_hand)
                        np.copyto(seq_input_buf, features['seq_input'])
                        np.copyto(glob_input_buf, features['glob_input'])
                    else:
                        seq_input_buf.fill(0)
                        glob_input_buf.fill(0)

                    rewards_buf[worker_id] = reward
                    dones_buf[worker_id] = done
                    np.copyto(actions_mask_buf, actions_mask)
                    glob_mask_buf.fill(0)
                    active_player_buf[worker_id] = prev_player
                    result_buf[worker_id] = -1
                    end_reason_buf[worker_id] = end_reason

                    remote.send('done')
                    continue

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

                if obs.current:
                    end_reason = get_end_reason(obs)
                    events = detect_events(old_state, obs.current, prev_player, obs.logs)
                    reward = calculate_step_reward(obs.current, prev_player, events, end_reason)
                    old_state = obs.current
                    done = (obs.current.result != -1)
                    active_p = prev_player
                    next_p = obs.current.yourIndex if obs.current else prev_player
                    turn_changed_buf[worker_id] = (active_p != next_p) and not done
                    
                    # Update opp_known_hand tracking
                    opp_index = 1 - prev_player # Assume prev_player is our index (yourIndex)
                    for log in obs.logs:
                        if log.type in [LogType.MOVE_CARD, LogType.DRAW]:
                            if getattr(log, 'toArea', None) == 2 and log.playerIndex == opp_index: # AreaType.HAND == 2
                                if log.cardId is not None and log.serial is not None:
                                    # Add to known hand if not already there
                                    if not any(c['serial'] == log.serial for c in opp_known_hand):
                                        opp_known_hand.append({'id': log.cardId, 'serial': log.serial})
                            
                            if getattr(log, 'fromArea', None) == 2 and log.playerIndex == opp_index:
                                if log.serial is not None:
                                    opp_known_hand = [c for c in opp_known_hand if c['serial'] != log.serial]
                                    
                        elif log.type in [LogType.PLAY, LogType.ATTACH, LogType.EVOLVE, LogType.DEVOLVE]:
                            if log.playerIndex == opp_index and log.serial is not None:
                                opp_known_hand = [c for c in opp_known_hand if c['serial'] != log.serial]
                    
                    # Truncate to 20
                    if len(opp_known_hand) > 20:
                        opp_known_hand = opp_known_hand[-20:]
                    
                    # Simpan result dan end_reason sebelum auto-reset
                    done_result = obs.current.result
                    done_end_reason = end_reason
                else:
                    active_p = prev_player
                    reward = -2.0
                    done = True
                    events = {}
                    
                    done_result = -1
                    done_end_reason = 0
                    turn_changed_buf[worker_id] = False

                if done:
                    game_step_counter = 0
                    reset_trackers()
                    opp_known_hand.clear()
                    battle_finish()
                    success = False
                    for _ in range(10):
                        try:
                            deck_list = [sample_deck(), sample_deck()]

                            obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                            obs = to_dataclass(obs_dict, Observation)
                            old_state = obs.current

                            if obs.current and obs.select and obs.current.result == -1:
                                features = extract_features(obs.current, obs.select, obs.current.yourIndex, opp_known_hand)
                                np.copyto(seq_input_buf, features['seq_input'])
                                np.copyto(glob_input_buf, features['glob_input'])
                                success = True
                                break
                        except Exception as e:
                            battle_finish()
                            
                    if not success:
                        print(f"[Worker {worker_id}] Auto-reset failed 10 times, using fallback deck.")
                        deck_list = [[1]*56 + [210]*4, [1]*56 + [210]*4]
                        try:
                            obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                            obs = to_dataclass(obs_dict, Observation)
                            old_state = obs.current
                            if obs.current and obs.select and obs.current.result == -1:
                                features = extract_features(obs.current, obs.select, obs.current.yourIndex, opp_known_hand)
                                np.copyto(seq_input_buf, features['seq_input'])
                                np.copyto(glob_input_buf, features['glob_input'])
                            else:
                                raise ValueError("Fallback deck failed")
                        except Exception as e:
                            print(f"[Worker {worker_id}] FATAL Auto-reset error: {e}")
                            obs = Observation(current=None, select=None, logs=[])
                            old_state = None
                            seq_input_buf.fill(0)
                            glob_input_buf.fill(0)
                else:
                    if obs.current and obs.select and obs.current.result == -1:
                        features = extract_features(obs.current, obs.select, obs.current.yourIndex, opp_known_hand)
                        np.copyto(seq_input_buf, features['seq_input'])
                        np.copyto(glob_input_buf, features['glob_input'])
                    else:
                        seq_input_buf.fill(0)
                        glob_input_buf.fill(0)

                rewards_buf[worker_id] = reward
                dones_buf[worker_id] = done
                np.copyto(actions_mask_buf, actions_mask)
                np.copyto(glob_mask_buf, legal_mask)
                active_player_buf[worker_id] = active_p
                result_buf[worker_id] = done_result if done else (obs.current.result if obs.current else -1)
                end_reason_buf[worker_id] = done_end_reason if done else 0

                remote.send('done')

            elif cmd == 'reset':
                game_step_counter = 0
                reset_trackers()
                battle_finish()
                success = False
                for _ in range(10):
                    try:
                        deck_list = [sample_deck(), sample_deck()]

                        obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                        obs = to_dataclass(obs_dict, Observation)
                        old_state = obs.current

                        if obs.current and obs.select and obs.current.result == -1:
                            features = extract_features(obs.current, obs.select, obs.current.yourIndex, opp_known_hand)
                            np.copyto(seq_input_buf, features['seq_input'])
                            np.copyto(glob_input_buf, features['glob_input'])
                            success = True
                            break
                    except Exception as e:
                        battle_finish()
                        
                if not success:
                    print(f"[Worker {worker_id}] Reset failed 10 times, using fallback deck.")
                    deck_list = [[1]*56 + [210]*4, [1]*56 + [210]*4]
                    try:
                        obs_dict, _ = battle_start(deck_list[0], deck_list[1])
                        obs = to_dataclass(obs_dict, Observation)
                        old_state = obs.current
                        if obs.current and obs.select and obs.current.result == -1:
                            features = extract_features(obs.current, obs.select, obs.current.yourIndex, opp_known_hand)
                            np.copyto(seq_input_buf, features['seq_input'])
                            np.copyto(glob_input_buf, features['glob_input'])
                        else:
                            raise ValueError("Fallback deck failed")
                    except Exception as e:
                        print(f"[Worker {worker_id}] FATAL Reset error: {e}")
                        obs = Observation(current=None, select=None, logs=[])
                        old_state = None
                        seq_input_buf.fill(0)
                        glob_input_buf.fill(0)
                    
                turn_changed_buf[worker_id] = False

                remote.send('done')

            elif cmd == 'close':
                battle_finish()
                remote.close()
                break

    except EOFError:
        pass
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        seq_input_buf.fill(0)
        glob_input_buf.fill(0)
        rewards_buf[worker_id] = -2.0
        dones_buf[worker_id] = True
        actions_mask_buf.fill(False)
        glob_mask_buf.fill(0)
        active_player_buf[worker_id] = 0
        turn_changed_buf[worker_id] = False
        result_buf[worker_id] = -1
        end_reason_buf[worker_id] = 0
        try:
            remote.send('done')
        except:
            pass
    finally:
        for shm in shms.values():
            shm.close()


class VectorEnv:
    """
    Manajer Lingkungan Paralel.
    P0 dan P1 selalu mendapat deck BERBEDA.
    Dilengkapi Shared Memory untuk eliminasi overhead Pipe.
    """
    def __init__(self, num_envs, new_deck_path="new_deck", gen_deck_path="agent_rl/deck_generated"):
        self.num_envs = num_envs
        
        # Validasi path
        def validate_path(d_path):
            if not os.path.exists(d_path):
                alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_generated")
                if os.path.exists(alt): return alt
                alt2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_rl", "deck_generated")
                if os.path.exists(alt2): return alt2
            return d_path
            
        new_deck_path = validate_path(new_deck_path)
        gen_deck_path = validate_path(gen_deck_path)

        self.shms = []
        self.shm_names = {}

        def create_shm(name, shape, dtype):
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = SharedMemory(create=True, size=size)
            self.shms.append(shm)
            self.shm_names[name] = shm.name
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            arr.fill(0)
            return arr

        self.seq_input = create_shm('seq_input', (num_envs, 113, 31), np.float32)
        self.glob_input = create_shm('glob_input', (num_envs, 266), np.float32)
        self.logits = create_shm('logits', (num_envs, 250), np.float32)
        self.rewards = create_shm('rewards', (num_envs,), np.float32)
        self.dones = create_shm('dones', (num_envs,), np.bool_)
        
        self.actions_mask = create_shm('actions_mask', (num_envs, 250), np.bool_)
        self.glob_mask = create_shm('glob_mask', (num_envs, 250), np.float32)
        self.active_player = create_shm('active_player', (num_envs,), np.int32)
        self.turn_changed = create_shm('turn_changed', (num_envs,), np.bool_)
        self.result = create_shm('result', (num_envs,), np.int32)
        self.end_reason = create_shm('end_reason', (num_envs,), np.int32)

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        ctx = mp.get_context('spawn')
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=worker, args=(work_remote, remote, i, new_deck_path, gen_deck_path, num_envs, self.shm_names))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send('reset')
        for remote in self.remotes:
            remote.recv()
            
        return {
            "seq_input": self.seq_input.copy(),
            "glob_input": self.glob_input.copy()
        }

    def step_async(self, logits_batch):
        np.copyto(self.logits, logits_batch)
        for remote in self.remotes:
            remote.send('step')

    def step_wait(self):
        for remote in self.remotes:
            remote.recv()
            
        batch_features = {
            "seq_input": self.seq_input.copy(),
            "glob_input": self.glob_input.copy()
        }
        
        rewards = self.rewards.copy()
        dones = self.dones.copy()
        
        infos = []
        for i in range(self.num_envs):
            infos.append({
                "actions_mask": self.actions_mask[i].copy(),
                "glob_mask": self.glob_mask[i].copy(),
                "active_player": self.active_player[i],
                "turn_changed": self.turn_changed[i],
                "result": self.result[i],
                "end_reason": self.end_reason[i]
            })
            
        return batch_features, rewards, dones, infos

    def step(self, logits_batch):
        self.step_async(logits_batch)
        return self.step_wait()

    def close(self):
        for remote in self.remotes:
            try:
                remote.send('close')
            except:
                pass
        for p in self.processes:
            p.join()
        for shm in self.shms:
            shm.close()
            shm.unlink()
