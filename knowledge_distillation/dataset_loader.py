import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from types import SimpleNamespace

import sys
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
        self._load_all_files()

    def _load_all_files(self):
        print(f"Memuat {len(self.files)} file dari {self.directory}...")
        for filepath in self.files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                rewards = data.get("rewards", [])
                if not rewards or len(rewards) < 2 or any(r is None for r in rewards):
                    continue
                    
                steps = data.get("steps", [])
                if not steps:
                    continue
                
                total_turns = len(steps)
                
                # Proses masing-masing pemain
                for player_idx in range(len(rewards)):
                    final_reward = float(rewards[player_idx])
                    
                    seq_inputs = np.zeros((self.max_steps, 173, 31), dtype=np.float32)
                    glob_inputs = np.zeros((self.max_steps, 266), dtype=np.float32)
                    action_masks = np.zeros((self.max_steps, 250), dtype=np.float32)
                    target_actions = np.zeros((self.max_steps,), dtype=np.int64)
                    target_values = np.zeros((self.max_steps,), dtype=np.float32)
                    valid_masks = np.zeros((self.max_steps,), dtype=np.float32)
                    
                    time_step = 0
                    
                    for step_idx, step_arr in enumerate(steps):
                        if time_step >= self.max_steps:
                            break # Maksimal limit step per ronde tercapai
                            
                        if len(step_arr) <= player_idx:
                            continue
                            
                        player_turn = step_arr[player_idx]
                        if not player_turn.get("action"):
                            continue
                        
                        action_data = player_turn.get("action", [])
                        if not action_data or not isinstance(action_data, list) or len(action_data) == 0:
                            continue
                        
                        # action_data is a flat list of integers, e.g. [0] or [4, 0]
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
                        
                        seq_inputs[time_step] = features["seq_input"]
                        glob_inputs[time_step] = features["glob_input"]
                        target_actions[time_step] = target_action
                        target_values[time_step] = target_value
                        action_masks[time_step] = features["glob_input"][16:16+250]
                        valid_masks[time_step] = 1.0
                        
                        time_step += 1
                        
                    # Simpan data jika pemain ini melakukan setidaknya 1 action di game ini
                    if time_step > 0:
                        self.samples.append({
                            "seq_input": seq_inputs,
                            "glob_input": glob_inputs,
                            "target_action": target_actions,
                            "target_value": target_values,
                            "action_mask": action_masks,
                            "valid_mask": valid_masks
                        })
            except Exception as e:
                import traceback
                traceback.print_exc()
                pass
                
        print(f"Berhasil mengumpulkan {len(self.samples)} sequence data (padded ke {self.max_steps}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["seq_input"], dtype=torch.float32),
            torch.tensor(sample["glob_input"], dtype=torch.float32),
            torch.tensor(sample["target_action"], dtype=torch.long),
            torch.tensor(sample["target_value"], dtype=torch.float32),
            torch.tensor(sample["action_mask"], dtype=torch.float32),
            torch.tensor(sample["valid_mask"], dtype=torch.float32)
        )
