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

def advance_to_player0(obs, params_opp, model_apply, decode_action, rng):
    """
    Memajukan simulasi secara otomatis jika giliran saat ini adalah milik Player 1.
    Player 1 dimainkan oleh Model AI Masa Lalu (Self-Play) menggunakan JAX di CPU.
    """
    from cg.game import battle_select
    from cg.api import to_dataclass, Observation, OptionType
    from agent_rl.feature_extractor import extract_features
    import jax
    import jax.numpy as jnp
    import random
    
    while obs.current is not None and obs.current.result == -1 and obs.current.yourIndex != 0:
        if obs.select is not None and obs.select.option:
            opt_count = len(obs.select.option)
            if params_opp is not None:
                # SELF PLAY MODE
                features = extract_features(obs.current, obs.select, 1)
                seq_input = np.expand_dims(features["seq_input"], axis=0)
                glob_input = np.expand_dims(features["glob_input"], axis=0)
                
                rng, step_rng = jax.random.split(rng)
                masked_logits, _ = model_apply(params_opp, seq_input, glob_input)
                
                # Gunakan gumbel softmax / categorical untuk sedikit variasi atau argmax langsung
                # Menggunakan argmax untuk performa stabil
                action_idx = int(jnp.argmax(masked_logits[0]))
                
                mock_select_dict = {"options": [{"type": OptionType(o.type).name, "index": o.index} for o in obs.select.option]}
                choices = decode_action(action_idx, mock_select_dict)
            else:
                # FALLBACK RANDOM BOT
                min_c = obs.select.minCount
                max_c = obs.select.maxCount
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
                try:
                    obs_dict = battle_select([0] if opt_count > 0 and obs.select.minCount > 0 else [])
                    obs = to_dataclass(obs_dict, Observation)
                except:
                    break
        else:
            break
            
    return obs, rng

def worker(remote, parent_remote, worker_id, deck_path, is_self_play):
    """
    Fungsi independen yang berjalan di sub-process untuk mengisolasi State C++ Engine.
    """
    # Pindahkan beban model lawan (Player 1) ke GPU kedua (GPU 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
    
    parent_remote.close()
    
    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_dataclass, Observation
    from agent_rl.feature_extractor import extract_features
    from agent_rl.reward import calc_potential, calculate_step_reward
    from agent_rl.action_mapping import decode_action
    
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from agent_rl.model import PokemonAgent
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
    old_potential = 0.0
    your_index = 0
    rng = jax.random.PRNGKey(worker_id)
    
    # Inisialisasi Model Player 1 (Frozen)
    params_opp = None
    model_apply = None
    if is_self_play:
        model = PokemonAgent(num_actions=250)
        model_apply = jax.jit(model.apply)
        dummy_seq = jnp.zeros((1, 93, 31))
        dummy_glob = jnp.zeros((1, 266))
        rng, init_rng = jax.random.split(rng)
        params_opp = model.init(init_rng, dummy_seq, dummy_glob)
        
        cp_path = "checkpoints/model_final.msgpack"
        if os.path.exists(cp_path):
            with open(cp_path, 'rb') as f:
                params_opp = serialization.from_bytes(params_opp, f.read())
    
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
                    
                # 2. Majukan game secara otomatis jika sekarang giliran Player 1
                obs, rng = advance_to_player0(obs, params_opp, model_apply, decode_action, rng)
                
                # 3. Hitung Reward setelah siklus selesai (kembali ke giliran Player 0 atau Game Over)
                if obs.current:
                    new_potential = calc_potential(obs.current, your_index)
                    reward = calculate_step_reward(old_potential, new_potential, obs.current, your_index)
                    old_potential = new_potential
                    done = (obs.current.result != -1)
                else:
                    reward = -1.0
                    done = True
                    
                info = {}
                
                # --- AUTO-RESET LOGIC ---
                if done:
                    info["turn"] = obs.current.turn if (obs and getattr(obs, 'current', None)) else 0
                    battle_finish()
                    try:
                        deck0 = random.choice(loaded_decks)
                        deck1 = random.choice(loaded_decks)
                        obs_dict, _ = battle_start(deck0, deck1)
                        obs = to_dataclass(obs_dict, Observation)
                        obs, rng = advance_to_player0(obs, params_opp, model_apply, decode_action, rng)
                        old_potential = calc_potential(obs.current, your_index) if obs.current else 0.0
                    except Exception as e:
                        print(f"Error during auto-reset: {e}")
                        obs = Observation(current=None, select=None, logs=[])
                        
                # 4. Ekstrak Fitur terbaru untuk state berikutnya
                if obs.current and obs.select and obs.current.result == -1:
                    features = extract_features(obs.current, obs.select, your_index)
                else:
                    features = empty_features
                    
                remote.send((features, reward, done, info))
                
            elif cmd == 'reset':
                battle_finish()
                deck0 = random.choice(loaded_decks)
                deck1 = random.choice(loaded_decks)
                obs_dict, _ = battle_start(deck0, deck1)
                obs = to_dataclass(obs_dict, Observation)
                
                # Biarkan Player 1 main duluan jika dia menang undian turn pertama
                obs, rng = advance_to_player0(obs, params_opp, model_apply, decode_action, rng)
                
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
            remote.send((empty_features, -1.0, True, {}))

class VectorEnv:
    """
    Manajer Lingkungan Paralel (Actor-Learner Bridge).
    Membungkus banyak environment (worker) agar JAX bisa memprosesnya secara batch.
    """
    def __init__(self, num_envs, deck_path="agent_rl/deck.csv", is_self_play=False):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        
        # Harus menggunakan mode 'spawn' agar pustaka C++ (SO/DLL) tidak berbenturan
        ctx = mp.get_context('spawn')
        
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=worker, args=(work_remote, remote, i, deck_path, is_self_play))
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
        infos = [res[3] for res in results]
        
        return batch_features, rewards, dones, infos

    def step(self, actions):
        """Convenience function (Gabungan async & wait)."""
        self.step_async(actions)
        return self.step_wait()
        
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
