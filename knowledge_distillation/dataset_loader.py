import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import gc
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataclasses
from cg.api import OptionType
from tcg_core.feature_extractor import extract_features
from tcg_core.action_mapping import get_action_index_for_option

original_asdict = dataclasses.asdict
def safe_asdict(obj, *args, **kwargs):
    if hasattr(obj, "__dict__") and not dataclasses.is_dataclass(obj):
        def _to_dict(o):
            if hasattr(o, "__dict__"):
                return {k: _to_dict(v) for k, v in o.__dict__.items()}
            elif isinstance(o, list):
                return [_to_dict(i) for i in o]
            return o
        return _to_dict(obj)
    return original_asdict(obj, *args, **kwargs)
dataclasses.asdict = safe_asdict

def dict_to_obj(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_obj(i) for i in d]
    else:
        return d

class KaggleReplayDataset(Dataset):
    def __init__(self, directory, max_files=None, gamma=0.99, max_steps=256):
        self.directory = directory
        self.files = glob.glob(os.path.join(directory, "*.json"))
        if max_files is not None:
            self.files = self.files[:max_files]
            
        self.gamma = gamma
        self.max_steps = max_steps
        self.samples = []
        self._preload_to_ram()

    def _preload_to_ram(self):
        print(f"Pre-loading & Ekstraksi Fitur {len(self.files)} file ke RAM (Mohon tunggu beberapa menit)...")
        from tqdm import tqdm
        
        for filepath in tqdm(self.files):
            try:
                with open(filepath, 'rb') as f:
                    data = json.loads(f.read())
            except Exception:
                continue
                
            rewards = data.get("rewards", [])
            if not rewards or len(rewards) < 2 or any(r is None for r in rewards):
                del data
                continue
                
            steps = data.get("steps", [])
            if not steps:
                del data
                continue
                
            total_turns = len(steps)
            
            for player_idx in [0, 1]:
                final_reward = float(rewards[player_idx])
                
                seq_list = []
                glob_list = []
                target_a_list = []
                target_v_list = []
                mask_list = []
                
                for step_idx, step_arr in enumerate(steps):
                    if len(step_arr) <= player_idx:
                        continue
                        
                    player_turn = step_arr[player_idx]
                    if not player_turn.get("action"):
                        continue
                    
                    action_data = player_turn.get("action", [])
                    if not action_data or not isinstance(action_data, list) or len(action_data) == 0:
                        continue
                    
                    chosen_idx = action_data[0]
                    obs = player_turn.get("observation", player_turn.get("obs", {}))
                    current_state = obs.get("current")
                    select_data_dict = obs.get("select")
                    
                    if not current_state or not select_data_dict:
                        continue
                        
                    options = select_data_dict.get("option", [])
                    if chosen_idx >= len(options):
                        continue
                        
                    chosen_option = options[chosen_idx].copy()
                    if "type" in chosen_option and isinstance(chosen_option["type"], int):
                        chosen_option["type"] = OptionType(chosen_option["type"]).name
                        
                    target_action = get_action_index_for_option(chosen_option, chosen_idx)
                    your_index = current_state.get("yourIndex", player_idx)
                    
                    steps_to_end = total_turns - step_idx - 1
                    target_value = final_reward * (self.gamma ** steps_to_end)
                    
                    mock_state = dict_to_obj(current_state)
                    mock_select = dict_to_obj(select_data_dict)
                    
                    features = extract_features(mock_state, mock_select, your_index)
                    
                    seq_list.append(features["seq_input"])
                    glob_list.append(features["glob_input"])
                    target_a_list.append(target_action)
                    target_v_list.append(target_value)
                    mask_list.append(features["glob_input"][16:16+250])
                    
                    if len(seq_list) >= self.max_steps:
                        break
                
                if len(seq_list) > 0:
                    self.samples.append({
                        "seq": np.stack(seq_list).astype(np.int16),
                        "glob": np.stack(glob_list).astype(np.int16),
                        "ta": np.array(target_a_list, dtype=np.int16),
                        "tv": np.array(target_v_list, dtype=np.float16),
                        "mask": np.stack(mask_list).astype(np.bool_)
                    })
            
            # Hapus data mentah json agar RAM efisien
            del data, steps
            
            # Paksa Garbage Collector berjalan secara periodik untuk menghindari penumpukan memori
            if len(self.samples) % 500 == 0:
                gc.collect()
            
        print(f"Selesai! Tercatat {len(self.samples)} kombinasi game player yang valid tersimpan di RAM.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        T = sample["seq"].shape[0]
        
        seq_inputs = np.zeros((self.max_steps, 173, 31), dtype=np.float32)
        glob_inputs = np.zeros((self.max_steps, 266), dtype=np.float32)
        action_masks = np.zeros((self.max_steps, 250), dtype=np.float32)
        target_actions = np.zeros((self.max_steps,), dtype=np.int64)
        target_values = np.zeros((self.max_steps,), dtype=np.float32)
        valid_masks = np.zeros((self.max_steps,), dtype=np.float32)
        
        seq_inputs[:T] = sample["seq"]
        glob_inputs[:T] = sample["glob"]
        target_actions[:T] = sample["ta"]
        target_values[:T] = sample["tv"]
        action_masks[:T] = sample["mask"]
        valid_masks[:T] = 1.0
        
        return (
            torch.tensor(seq_inputs, dtype=torch.float32),
            torch.tensor(glob_inputs, dtype=torch.float32),
            torch.tensor(target_actions, dtype=torch.long),
            torch.tensor(target_values, dtype=torch.float32),
            torch.tensor(action_masks, dtype=torch.float32),
            torch.tensor(valid_masks, dtype=torch.float32)
        )
