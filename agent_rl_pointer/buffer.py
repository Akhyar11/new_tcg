import numpy as np

class RolloutBuffer:
    """
    Memori Penyimpanan Pengalaman (Rollout Buffer) untuk PPO + LSTM (TBPTT).
    Menyimpan hidden state (carry) awal setiap langkah dan mengembalikan potongan
    urutan waktu (sequences) saat iterasi batch.
    """
    def __init__(self, n_steps, num_envs, seq_shape=(173, 31), glob_shape=(266,)):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.buffer_size = n_steps * num_envs
        self.step = 0

        # Pre-alokasi array memori
        self.seq_inputs = np.zeros((n_steps, num_envs, *seq_shape), dtype=np.float32)
        self.glob_inputs = np.zeros((n_steps, num_envs, *glob_shape), dtype=np.float32)
        self.actions_mask = np.zeros((n_steps, num_envs, 250), dtype=np.bool_)
        self.log_probs = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.turn_changed = np.zeros((n_steps, num_envs), dtype=np.bool_)
        self.active_players = np.zeros((n_steps, num_envs), dtype=np.int32)
        
        # LSTM Carry States (c, h) untuk setiap awal step
        # Default hidden_size kita adalah 256
        self.carry_c = np.zeros((n_steps, num_envs, 256), dtype=np.float32)
        self.carry_h = np.zeros((n_steps, num_envs, 256), dtype=np.float32)

    def add(self, seq_in, glob_in, actions_mask, log_prob, reward, value, done, turn_changed, active_player, carry_c, carry_h):
        if self.step >= self.n_steps:
            raise ValueError("Buffer sudah penuh, jalankan compute_returns dan clear() terlebih dahulu.")

        self.seq_inputs[self.step] = np.array(seq_in, copy=False)
        self.glob_inputs[self.step] = np.array(glob_in, copy=False)
        self.actions_mask[self.step] = np.array(actions_mask, copy=False)
        self.log_probs[self.step] = np.array(log_prob, copy=False)
        self.rewards[self.step] = np.array(reward, copy=False)
        self.values[self.step] = np.array(value, copy=False)
        self.dones[self.step] = np.array(done, dtype=np.float32, copy=False)
        self.turn_changed[self.step] = np.array(turn_changed, dtype=np.bool_, copy=False)
        self.active_players[self.step] = np.array(active_player, dtype=np.int32, copy=False)
        self.carry_c[self.step] = np.array(carry_c, dtype=np.float32, copy=False)
        self.carry_h[self.step] = np.array(carry_h, dtype=np.float32, copy=False)

        self.step += 1

    def compute_returns_and_advantages(self, last_values, last_dones, gamma=0.99, gae_lambda=0.95):
        self.returns = np.zeros_like(self.rewards)
        self.advantages = np.zeros_like(self.rewards)

        last_gae_lam = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                actual_next_values = np.where(self.turn_changed[t], -last_values, last_values)
            else:
                next_non_terminal = 1.0 - self.dones[t]
                actual_next_values = np.where(self.turn_changed[t], -self.values[t + 1], self.values[t + 1])

            delta = self.rewards[t] + gamma * actual_next_values * next_non_terminal - self.values[t]
            
            actual_last_gae_lam = np.where(self.turn_changed[t], -last_gae_lam, last_gae_lam)
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * actual_last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size, seq_len=32):
        """
        Menghasilkan mini-batch berbentuk Sequence: (num_seqs, seq_len, ...)
        batch_size disini adalah total jumlah transition.
        Jumlah sekuens per batch (num_seqs) = batch_size // seq_len
        """
        if self.step < self.n_steps:
            raise ValueError("Tidak dapat mengambil batch karena buffer belum penuh.")
            
        assert self.n_steps % seq_len == 0, f"n_steps ({self.n_steps}) harus habis dibagi seq_len ({seq_len})"
        assert batch_size % seq_len == 0, f"batch_size ({batch_size}) harus habis dibagi seq_len ({seq_len})"
        
        num_seqs_per_env = self.n_steps // seq_len
        total_seqs = num_seqs_per_env * self.num_envs
        num_seqs_per_batch = batch_size // seq_len
        
        # 1. Reshape menjadi (num_seqs_per_env, seq_len, num_envs, ...)
        # 2. Swap axis ke (num_seqs_per_env, num_envs, seq_len, ...)
        # 3. Flatten 2 axis pertama jadi (total_seqs, seq_len, ...)
        def _to_seq(arr):
            s = arr.reshape(num_seqs_per_env, seq_len, self.num_envs, *arr.shape[2:])
            s = s.swapaxes(1, 2)
            return s.reshape(total_seqs, seq_len, *arr.shape[2:])
            
        b_seq = _to_seq(self.seq_inputs)
        b_glob = _to_seq(self.glob_inputs)
        b_actions_mask = _to_seq(self.actions_mask)
        b_log_probs = _to_seq(self.log_probs)
        b_advantages = _to_seq(self.advantages)
        b_returns = _to_seq(self.returns)
        b_values = _to_seq(self.values)
        b_active_players = _to_seq(self.active_players)
        
        # Carry hanya dibutuhkan PADA AWAL sequence (t=0)
        # Jadi shape nya (total_seqs, num_features)
        b_carry_c_all = _to_seq(self.carry_c) # (total_seqs, seq_len, 256)
        b_carry_h_all = _to_seq(self.carry_h)
        b_carry_c = b_carry_c_all[:, 0, :]    # Ambil index 0 (awal seq)
        b_carry_h = b_carry_h_all[:, 0, :]

        # Normalisasi Advantages HANYA untuk P0 (active_player == 0)
        p0_mask = (b_active_players == 0)
        if p0_mask.sum() > 0:
            p0_mean = b_advantages[p0_mask].mean()
            p0_std = b_advantages[p0_mask].std() + 1e-8
            b_advantages = np.where(p0_mask, (b_advantages - p0_mean) / p0_std, b_advantages)

        # Acak urutan SEQUENCE (bukan step individual)
        indices = np.random.permutation(total_seqs)

        start_idx = 0
        while start_idx < total_seqs:
            end_idx = min(start_idx + num_seqs_per_batch, total_seqs)
            batch_indices = indices[start_idx:end_idx]

            yield {
                "seq_input": b_seq[batch_indices],
                "glob_input": b_glob[batch_indices],
                "actions_mask": b_actions_mask[batch_indices],
                "old_log_probs": b_log_probs[batch_indices],
                "advantages": b_advantages[batch_indices],
                "returns": b_returns[batch_indices],
                "values": b_values[batch_indices],
                "active_players": b_active_players[batch_indices],
                "carry_c": b_carry_c[batch_indices],
                "carry_h": b_carry_h[batch_indices]
            }
            start_idx = end_idx

    def clear(self):
        self.step = 0
