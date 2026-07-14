import numpy as np


class RolloutBuffer:
    """
    Memori Penyimpanan Pengalaman (Rollout Buffer) untuk PPO.
    Berbasis NumPy murni untuk kecepatan pre-alokasi dan menghindari overhead list Python.
    Memori dialokasikan di awal (Pre-allocated) dengan dimensi (n_steps, num_envs, ...).
    """
    def __init__(self, n_steps, num_envs, seq_shape=(113, 31), glob_shape=(266,)):
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

    def add(self, seq_in, glob_in, actions_mask, log_prob, reward, value, done, turn_changed, active_player):
        """
        Menyimpan data hasil dari satu step ke dalam buffer.
        Semua input berdimensi (num_envs, ...).
        """
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

        self.step += 1

    def compute_returns_and_advantages(self, last_values, last_dones, gamma=0.99, gae_lambda=0.95):
        """
        Menghitung Generalized Advantage Estimation (GAE) untuk seluruh data di buffer.
        """
        self.returns = np.zeros_like(self.rewards)
        self.advantages = np.zeros_like(self.rewards)

        last_gae_lam = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                actual_next_values = np.where(self.turn_changed[t], -last_values, last_values)
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                actual_next_values = np.where(self.turn_changed[t], -self.values[t + 1], self.values[t + 1])

            # GAE Standard with Zero-Sum Correction
            delta = self.rewards[t] + gamma * actual_next_values * next_non_terminal - self.values[t]
            
            actual_last_gae_lam = np.where(self.turn_changed[t], -last_gae_lam, last_gae_lam)
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * actual_last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """
        Menghasilkan mini-batch data secara acak dari buffer yang telah diratakan (flattened).
        Digunakan pada fase pembaruan (PPO update).
        """
        if self.step < self.n_steps:
            raise ValueError("Tidak dapat mengambil batch karena buffer belum penuh.")

        # Mengubah bentuk (n_steps, num_envs, ...) menjadi (n_steps * num_envs, ...)
        b_seq = self.seq_inputs.reshape((self.buffer_size, -1, 31))
        b_glob = self.glob_inputs.reshape((self.buffer_size, -1))
        b_actions_mask = self.actions_mask.reshape((self.buffer_size, 250))
        b_log_probs = self.log_probs.reshape((self.buffer_size,))
        b_advantages = self.advantages.reshape((self.buffer_size,))
        b_returns = self.returns.reshape((self.buffer_size,))
        b_values = self.values.reshape((self.buffer_size,))
        b_active_players = self.active_players.reshape((self.buffer_size,))

        # Normalisasi Advantages di tingkat batch raksasa (membantu stabilitas JAX)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Acak urutan indeks
        indices = np.random.permutation(self.buffer_size)

        start_idx = 0
        while start_idx < self.buffer_size:
            end_idx = min(start_idx + batch_size, self.buffer_size)
            batch_indices = indices[start_idx:end_idx]

            yield {
                "seq_input": b_seq[batch_indices],
                "glob_input": b_glob[batch_indices],
                "actions_mask": b_actions_mask[batch_indices],
                "old_log_probs": b_log_probs[batch_indices],
                "advantages": b_advantages[batch_indices],
                "returns": b_returns[batch_indices],
                "values": b_values[batch_indices],
                "active_players": b_active_players[batch_indices]
            }
            start_idx = end_idx

    def clear(self):
        """Mengosongkan penunjuk langkah untuk pengumpulan rollout baru."""
        self.step = 0
