import multiprocessing as mp
import numpy as np
import random
import os

def load_deck(filepath):
    deck = []
    if not os.path.exists(filepath):
        # Fallback dummy deck if file missing
        return [1]*56 + [210]*4
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
    return deck

# Fungsi advance_to_player0 telah dihapus karena kita menggunakan True Self-Play.

def worker(remote, parent_remote, worker_id, deck_path):
    """
    Fungsi independen yang berjalan di sub-process untuk mengisolasi State C++ Engine.
    """
    parent_remote.close()
    
    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_dataclass, Observation
    from agent_rl.feature_extractor import extract_features
    from agent_rl.reward import calc_potential, calculate_step_reward
    from agent_rl.action_mapping import decode_action, get_action_index_for_option
    import glob
    
    # Pre-load semua deck dari folder atau file tunggal
    deck_files = []
    if os.path.isdir(deck_path):
        deck_files = glob.glob(os.path.join(deck_path, "*.csv"))
    else:
        deck_files = [deck_path]
        
    if not deck_files:
        print(f"Peringatan: Tidak ada deck ditemukan di {deck_path}. Menggunakan deck dummy.")
        loaded_decks = [[1]*56 + [210]*4]
    else:
        loaded_decks = [load_deck(f) for f in deck_files]
    
    obs = None
    old_potentials = {0: 0.0, 1: 0.0}
    
    empty_features = {
        "seq_input": np.zeros((93, 31), dtype=np.float32), 
        "glob_input": np.zeros(266, dtype=np.float32)
    }

    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                action_idx, top_actions_list = data
                
                # Buat mock_select_dict dari obs.select untuk decode_action
                from cg.api import OptionType
                mock_select_dict = {"options": []}
                min_c = 1
                if obs and obs.select and obs.select.option:
                    mock_select_dict = {"options": [{"type": OptionType(o.type).name, "index": o.index} for o in obs.select.option]}
                    min_c = obs.select.minCount
                
                # Pastikan action_idx utama yang terpilih (sampled action) ada di paling depan list
                if action_idx in top_actions_list:
                    top_actions_list.remove(action_idx)
                top_actions_list.insert(0, action_idx)
                
                # 1. Player 0 melakukan aksi (Decode JAX -> C++)
                choices = decode_action(top_actions_list, mock_select_dict, min_c)
                
                # Buat mask multi-hot dari pilihan yang benar-benar dieksekusi (untuk PPO Gradient)
                actions_mask = np.zeros(250, dtype=np.bool_)
                if mock_select_dict["options"]:
                    for c in choices:
                        if c < len(mock_select_dict["options"]):
                            actions_mask[get_action_index_for_option(mock_select_dict["options"][c])] = True
                
                if not np.any(actions_mask):
                    actions_mask[action_idx] = True
                
                try:
                    obs_dict = battle_select(choices)
                    obs = to_dataclass(obs_dict, Observation)
                except Exception as e:
                    # Coba fallback bot jika AI gagal parse
                    try:
                        opt_count = len(obs.select.option) if obs.select and obs.select.option else 0
                        min_c = obs.select.minCount if obs.select else 0
                        fallback_choices = list(range(min(opt_count, min_c))) if min_c > 0 else []
                        obs_dict = battle_select(fallback_choices)
                        obs = to_dataclass(obs_dict, Observation)
                    except:
                        # Jika error parah, paksa game over (penalti kalah)
                        obs = Observation(current=None, select=None, logs=[])
                    
                # Tidak ada lagi advance_to_player0 (AI mengendalikan kedua belah pihak)
                
                # 3. Hitung Reward setelah siklus selesai
                if obs.current:
                    active_p = obs.current.yourIndex
                    new_potential = calc_potential(obs.current, active_p)
                    # Ambil old_potential dari pemain yang sedang aktif
                    reward = calculate_step_reward(old_potentials[active_p], new_potential, obs.current, active_p)
                    old_potentials[active_p] = new_potential
                    done = (obs.current.result != -1)
                else:
                    active_p = 0
                    reward = -1.0
                    done = True
                    
                info = {"actions_mask": actions_mask, "active_player": active_p, "result": obs.current.result if obs.current else -1}
                    
                # --- AUTO-RESET LOGIC ---
                if done:
                    battle_finish()
                    try:
                        deck0 = random.choice(loaded_decks)
                        deck1 = random.choice(loaded_decks)
                        obs_dict, _ = battle_start(deck0, deck1)
                        obs = to_dataclass(obs_dict, Observation)
                        old_potentials = {0: calc_potential(obs.current, 0) if obs.current else 0.0, 
                                          1: calc_potential(obs.current, 1) if obs.current else 0.0}
                        
                        if obs.current and obs.select and obs.current.result == -1:
                            features = extract_features(obs.current, obs.select, obs.current.yourIndex)
                        else:
                            features = empty_features
                    except Exception as e:
                        print(f"Error during auto-reset: {e}")
                        obs = Observation(current=None, select=None, logs=[])
                        features = empty_features
                else:
                    # 4. Ekstrak Fitur terbaru untuk state berikutnya
                    if obs.current and obs.select and obs.current.result == -1:
                        features = extract_features(obs.current, obs.select, obs.current.yourIndex)
                    else:
                        features = empty_features
                    
                remote.send((features, reward, done, info))
                
            elif cmd == 'reset':
                battle_finish()
                deck0 = random.choice(loaded_decks)
                deck1 = random.choice(loaded_decks)
                obs_dict, _ = battle_start(deck0, deck1)
                obs = to_dataclass(obs_dict, Observation)
                
                old_potentials = {0: calc_potential(obs.current, 0) if obs.current else 0.0, 
                                  1: calc_potential(obs.current, 1) if obs.current else 0.0}
                
                if obs.current and obs.select and obs.current.result == -1:
                    features = extract_features(obs.current, obs.select, obs.current.yourIndex)
                else:
                    features = empty_features
                    
                remote.send(features)
                
            elif cmd == 'close':
                battle_finish()
                remote.close()
                break
                
        except EOFError:
            break
        except Exception as e:
            print(f"[Worker {worker_id}] Terjadi kesalahan internal: {e}")
            remote.send((empty_features, -1.0, True, {}))

class VectorEnv:
    """
    Manajer Lingkungan Paralel (Actor-Learner Bridge).
    Membungkus banyak environment (worker) agar JAX bisa memprosesnya secara batch.
    """
    def __init__(self, num_envs, deck_path="agent_rl/deck.csv"):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        
        # Harus menggunakan mode 'spawn' agar pustaka C++ (SO/DLL) tidak berbenturan
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
            
        # Mengumpulkan hasil dari semua worker
        results = [remote.recv() for remote in self.remotes]
        
        # Batching list of dicts menjadi dict of batched numpy arrays
        seq_inputs = np.stack([res["seq_input"] for res in results])
        glob_inputs = np.stack([res["glob_input"] for res in results])
        return {"seq_input": seq_inputs, "glob_input": glob_inputs}

    def step_async(self, actions, top_actions):
        """Mendistribusikan array action JAX ke masing-masing worker."""
        for remote, action, top_action in zip(self.remotes, actions, top_actions):
            remote.send(('step', (action, top_action.tolist())))

    def step_wait(self):
        """Menunggu eksekusi C++ selesai dan menggabungkan hasilnya menjadi format JAX."""
        results = [remote.recv() for remote in self.remotes]
        
        seq_inputs = np.stack([res[0]["seq_input"] for res in results])
        glob_inputs = np.stack([res[0]["glob_input"] for res in results])
        batch_features = {"seq_input": seq_inputs, "glob_input": glob_inputs}
        
        rewards = np.array([res[1] for res in results], dtype=np.float32)
        dones = np.array([res[2] for res in results], dtype=np.bool_)
        infos = [res[3] for res in results]
        
        return batch_features, rewards, dones, infos

    def step(self, actions, top_actions):
        """Convenience function (Gabungan async & wait)."""
        self.step_async(actions, top_actions)
        return self.step_wait()
        
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
