import jax
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import optax
from functools import partial

# Definisi fungsi Update murni (Pure Function) untuk JAX XLA Compilation
# Menggunakan partial untuk menandai fungsi apply sebagai argumen statis agar bisa di-JIT
@partial(jax.pmap, in_axes=(0, 0, 0, None, None, None, None), static_broadcasted_argnums=(3, 4), axis_name='gpu')
def ppo_update_step(params, opt_state, batch, apply_fn, tx, clip_ratio, entropy_coef=0.01):
    """
    Satu langkah optimasi (Gradient Descent) menggunakan algoritma PPO.
    Seluruh perhitungan di fungsi ini berjalan murni di dalam GPU/TPU via XLA.

    Args:
        entropy_coef: Koefisien entropy bonus (bisa di-anneal).
    """

    def loss_fn(p):
        # 1. Forward Pass (Mendapatkan Logits dan Value)
        # Bentuk logits: (Batch, 250), Bentuk values: (Batch, 1)
        logits, values = apply_fn(p, batch['seq_input'], batch['glob_input'])
        values = values.squeeze(-1) # ratakan menjadi (Batch,)

        # 2. Action Masking
        # Mask berada di indeks 16 sampai 265 pada glob_input
        action_mask = batch['glob_input'][..., 16:266]
        # Beri penalti tak terhingga pada opsi ilegal agar probabilitasnya mutlak 0
        masked_logits = logits + ((1.0 - action_mask) * -1e9)

        # 3. Hitung Log Probabilities
        log_probs_all = jax.nn.log_softmax(masked_logits)

        # Ambil log_prob dari semua aksi yang dipilih (Multi-Hot / Multi-Choice)
        # Gradient akan mengalir proporsional ke semua aksi yang terpilih pada turn ini
        mask_count = jnp.maximum(1.0, jnp.sum(batch['actions_mask'], axis=-1))
        log_probs = jnp.sum(log_probs_all * batch['actions_mask'], axis=-1) / mask_count

        # 4. PPO Actor Loss (Clipped Surrogate Objective)
        # ⚠️ Log-ratio CLIPPING di [-10, +10] → ratio maksimal exp(10)≈22026
        # Tanpa clipping, kalau old_log_probs = -50 dan log_probs ≈ 0,
        # ratio = exp(50) = 5e21 → PPO clip tidak bisa menyelamatkan → LOSS LEDAKAN
        log_ratio = jnp.clip(log_probs - batch['old_log_probs'], -10.0, 10.0)
        ratio = jnp.exp(log_ratio)

        # Active player mask (0 = P0, 1 = P1). Kita hanya train P0.
        active_mask = (batch['active_players'] == 0).astype(jnp.float32)
        valid_count = jnp.maximum(1.0, active_mask.sum())

        surr1 = ratio * batch['advantages'] * active_mask
        surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch['advantages'] * active_mask
        actor_loss = -jnp.minimum(surr1, surr2).sum() / valid_count

        # 5. PPO Critic Loss (Mean Squared Error dari TD-Returns)
        value_loss = 0.5 * (jnp.square(values - batch['returns']) * active_mask).sum() / valid_count

        # 6. Entropy Bonus (Untuk mendorong eksplorasi)
        probs = jax.nn.softmax(masked_logits)
        # Sum(P * logP) hanya untuk aksi yang legal
        entropy = -jnp.sum(probs * log_probs_all * action_mask, axis=-1)
        entropy = (entropy * active_mask).sum() / valid_count

        # 7. Total Loss Kombinasi
        total_loss = actor_loss + (0.5 * value_loss) - (entropy_coef * entropy)

        # Kembalikan tuple (Loss untuk gradient, dan metrik untuk logistik/monitoring)
        return total_loss, (actor_loss, value_loss, entropy)

    # Hitung nilai Loss dan Gradient secara sekuensial
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_metrics), grads = grad_fn(params)

    # Sinkronisasi / Rata-rata Gradient di seluruh GPU (multi-GPU)
    grads = jax.lax.pmean(grads, axis_name='gpu')
    loss = jax.lax.pmean(loss, axis_name='gpu')

    # Hitung pembaruan optimasi menggunakan Optax (misal: Adam)
    updates, new_opt_state = tx.update(grads, opt_state, params)

    # Aplikasikan pembaruan ke bobot (weights)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss, aux_metrics

@partial(jax.pmap, static_broadcasted_argnums=(1,), axis_name='gpu')
def get_action_and_value(params, apply_fn, seq_input, glob_input, key):
    """
    Fungsi inferensi untuk digunakan saat Rollout.
    Return raw logits (belum di-mask) — masking final dilakukan di train.py agar
    konsisten dengan cara masking di ppo_update_step.loss_fn.
    """
    logits, values = apply_fn(params, seq_input, glob_input)

    # Action Masking (hanya untuk sampling & log_prob)
    action_mask = glob_input[..., 16:266]
    masked_logits = logits + ((1.0 - action_mask) * -1e9)

    # Distribusi probabilitas (Categorical)
    actions = jax.random.categorical(key, masked_logits, axis=-1)

    # Ambil log_prob dari aksi yang dipilih
    log_probs_all = jax.nn.log_softmax(masked_logits)
    log_probs = jnp.take_along_axis(log_probs_all, actions[..., None], axis=-1).squeeze(-1)

    # Return RAW logits — masking di train.py untuk old_log_probs yang konsisten
    # dengan loss_fn (single mask, bukan double mask).
    return actions, log_probs, values.squeeze(-1), logits
