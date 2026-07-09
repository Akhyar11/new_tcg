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

def advance_to_player0(obs):
    """
    Memajukan simulasi secara otomatis jika giliran saat ini adalah milik Player 1.
    Player 1 akan dimainkan oleh Random Bot murni sampai giliran kembali ke Player 0,
    atau game berakhir.
    """
    from cg.game import battle_select
    from cg.api import to_dataclass, Observation
    
    while obs.current is not None and obs.current.result == -1 and obs.current.yourIndex != 0:
        if obs.select is not None:
            min_c = obs.select.minCount
            max_c = obs.select.maxCount
            opt_count = len(obs.select.option)
            choices = []
            
            if min_c > 0:
                max_c = min(max_c, opt_count)
                max_c = max(max_c, min_c)
                pick_count = random.randint(min_c, max_c)
                choices = random.sample(range(opt_count), pick_count)
            else:
                if opt_count > 0 and random.random() > 0.5:
                    pick_count = random.randint(1, min(max_c, opt_count))
                    choices = random.sample(range(opt_count), pick_count)
            
            try:
                obs_dict = battle_select(choices)
                obs = to_dataclass(obs_dict, Observation)
            except Exception as e:
                # Failsafe jika aksi random menyebabkan crash C++
                try:
                    obs_dict = battle_select([0] if opt_count > 0 and min_c > 0 else [])
                    obs = to_dataclass(obs_dict, Observation)
                except:
                    break
        else:
            break
            
    return obs

def worker(remote, parent_remote, worker_id, deck_path):
    """
    Fungsi independen yang berjalan di sub-process untuk mengisolasi State C++ Engine.
    """
    parent_remote.close()
    
    # Import diletakkan di dalam worker agar instance C++ DLL dimuat secara independen per proses
    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_dataclass, Observation
    from agent_rl.feature_extractor import extract_features
    from agent_rl.reward import calc_potential, calculate_step_reward
    from agent_rl.action_mapping import decode_action
    
    deck = load_deck(deck_path)
    obs = None
    old_potential = 0.0
    your_index = 0
    
    empty_features = {
        "seq_input": np.zeros((93, 31), dtype=np.float32), 
        "glob_input": np.zeros(266, dtype=np.float32)
    }

    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                action_idx = data
                
                # Buat mock_select_dict dari obs.select untuk decode_action
                from cg.api import OptionType
                mock_select_dict = {"options": []}
                if obs and obs.select and obs.select.option:
                    mock_select_dict = {"options": [{"type": OptionType(o.type).name, "index": o.index} for o in obs.select.option]}
                
                # 1. Player 0 melakukan aksi (Decode JAX -> C++)
                choices = decode_action(action_idx, mock_select_dict)
                
                try:
                    obs_dict = battle_select(choices)
                    obs = to_dataclass(obs_dict, Observation)
                except Exception as e:
                    # Jika error parah, paksa game over (penalti kalah)
                    obs = Observation(current=None, select=None, logs=[])
                    
                # 2. Majukan game secara otomatis jika sekarang giliran Player 1 (Random Bot)
                obs = advance_to_player0(obs)
                
                # 3. Hitung Reward setelah siklus selesai (kembali ke giliran Player 0 atau Game Over)
                if obs.current:
                    new_potential = calc_potential(obs.current, your_index)
                    reward = calculate_step_reward(old_potential, new_potential, obs.current, your_index)
                    old_potential = new_potential
                    done = (obs.current.result != -1)
                else:
                    reward = -1.0
                    done = True
                    
                # 4. Ekstrak Fitur terbaru untuk state berikutnya
                if not done and obs.current and obs.select:
                    features = extract_features(obs.current, obs.select, your_index)
                else:
                    features = empty_features
                    
                remote.send((features, reward, done))
                
            elif cmd == 'reset':
                battle_finish()
                obs_dict, _ = battle_start(deck, deck)
                obs = to_dataclass(obs_dict, Observation)
                
                # Biarkan Random Bot (Player 1) main duluan jika dia menang undian turn pertama
                obs = advance_to_player0(obs)
                
                old_potential = calc_potential(obs.current, your_index) if obs.current else 0.0
                
                if obs.current and obs.select and obs.current.result == -1:
                    features = extract_features(obs.current, obs.select, your_index)
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
            remote.send((empty_features, -1.0, True))

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

    def step_async(self, actions):
        """Mendistribusikan array action JAX ke masing-masing worker."""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        """Menunggu eksekusi C++ selesai dan menggabungkan hasilnya menjadi format JAX."""
        results = [remote.recv() for remote in self.remotes]
        
        seq_inputs = np.stack([res[0]["seq_input"] for res in results])
        glob_inputs = np.stack([res[0]["glob_input"] for res in results])
        batch_features = {"seq_input": seq_inputs, "glob_input": glob_inputs}
        
        rewards = np.array([res[1] for res in results], dtype=np.float32)
        dones = np.array([res[2] for res in results], dtype=np.bool_)
        
        return batch_features, rewards, dones

    def step(self, actions):
        """Convenience function (Gabungan async & wait)."""
        self.step_async(actions)
        return self.step_wait()
        
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
