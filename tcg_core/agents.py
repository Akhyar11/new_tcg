import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import dataclasses

from cg.api import OptionType
from tcg_core.feature_extractor import extract_features
import tcg_core.action_mapping as action_mapping

class BaseAgent:
    def __init__(self, name, model_cls, action_mapping_module, checkpoint_path=None):
        self.name = name
        self.model = model_cls(num_actions=250)
        self.get_action_idx = action_mapping_module.get_action_index_for_option
        self.create_action_mask = action_mapping_module.create_action_mask
        
        self.rng = jax.random.PRNGKey(42)
        self.params = self._init_params()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                self.params = serialization.from_bytes(self.params, f.read())
                
        self.apply_fn = jax.jit(self.model.apply)
        
    def _init_params(self):
        raise NotImplementedError
        
    def reset(self):
        """Reset internal states like LSTM carry. Override in subclasses if needed."""
        pass

    def softmax(self, x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / (exp_x.sum() + 1e-10)

    def get_choices_from_logits(self, logits_np, obs):
        options = obs.select.option
        min_c = obs.select.minCount
        max_c = obs.select.maxCount
        
        mock_options = []
        for o in options:
            d = dataclasses.asdict(o)
            d["type"] = OptionType(o.type).name
            mock_options.append(d)
        mock_select = {"options": mock_options}

        mask_array = self.create_action_mask(mock_select, min_c, max_c)
        masked = logits_np - 1e9 * (1.0 - mask_array)
        probs = self.softmax(masked)

        sampled_indices = []
        if probs.sum() > 0:
            remaining = probs.copy()
            for _ in range(max_c):
                if remaining.sum() <= 0: break
                p = remaining / remaining.sum()
                idx = int(np.random.choice(len(p), p=p))
                if idx == 160: # END Action
                    has_end_option = any(self.get_action_idx(opt, i) == 160 for i, opt in enumerate(mock_select["options"]))
                    if has_end_option:
                        sampled_indices.append(idx)
                        remaining[idx] = 0.0
                    elif len(sampled_indices) >= min_c: 
                        break
                    else:
                        remaining[idx] = 0.0
                        continue
                else:
                    sampled_indices.append(idx)
                    remaining[idx] = 0.0
        else:
            sampled_indices = [160]

        choices = []
        for jax_idx in sampled_indices:
            for cpp_idx, opt in enumerate(mock_select["options"]):
                mapped_idx = self.get_action_idx(opt, cpp_idx)
                if mapped_idx == jax_idx and cpp_idx not in choices:
                    choices.append(cpp_idx)
                    break

        if len(choices) < min_c:
            for cpp_idx in range(len(options)):
                if cpp_idx not in choices: choices.append(cpp_idx)
                if len(choices) >= min_c: break

        return choices
        
    def select_action(self, obs):
        raise NotImplementedError


class FFAgent(BaseAgent):
    def _init_params(self):
        dummy_seq = jnp.zeros((1, 173, 31))
        dummy_glob = jnp.zeros((1, 266))
        return self.model.init(self.rng, dummy_seq, dummy_glob)

    def select_action(self, obs):
        if not obs.select or not obs.select.option:
            return []
            
        features = extract_features(obs.current, obs.select, obs.current.yourIndex)
        seq_input = np.expand_dims(features["seq_input"], axis=0)
        glob_input = np.expand_dims(features["glob_input"], axis=0)

        logits_raw, _ = self.apply_fn(self.params, seq_input, glob_input)
        logits_np = np.array(logits_raw[0])
        
        return self.get_choices_from_logits(logits_np, obs)


class LSTMAgent(BaseAgent):
    def _init_params(self):
        dummy_seq = jnp.zeros((1, 173, 31))
        dummy_glob = jnp.zeros((1, 266))
        dummy_carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
        self.carry = dummy_carry
        return self.model.init(self.rng, dummy_seq, dummy_glob, dummy_carry)
        
    def reset(self):
        """Reset the LSTM hidden state (carry) at the start of a new game."""
        self.carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))

    def select_action(self, obs):
        if not obs.select or not obs.select.option:
            return []
            
        features = extract_features(obs.current, obs.select, obs.current.yourIndex)
        seq_input = np.expand_dims(features["seq_input"], axis=0)
        glob_input = np.expand_dims(features["glob_input"], axis=0)

        logits_raw, values, new_carry = self.apply_fn(self.params, seq_input, glob_input, self.carry)
        self.carry = new_carry
        self.last_value = float(values[0][0])
        logits_np = np.array(logits_raw[0])
        
        return self.get_choices_from_logits(logits_np, obs)
