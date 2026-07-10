import jax
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import optax
from functools import partial

# Definisi fungsi Update murni (Pure Function) untuk JAX XLA Compilation
# Menggunakan partial untuk menandai fungsi apply sebagai argumen statis agar bisa di-JIT
@partial(jax.jit, static_argnames=['apply_fn', 'tx'])
def ppo_update_step(params, opt_state, batch, apply_fn, tx, clip_ratio=0.2, val_coef=0.5, ent_coef=0.01):
    """
    Satu langkah optimasi (Gradient Descent) menggunakan algoritma PPO.
    Seluruh perhitungan di fungsi ini berjalan murni di dalam GPU/TPU via XLA.
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
        log_probs = jnp.sum(log_probs_all * batch['actions_mask'], axis=-1)
        
        # 4. PPO Actor Loss (Clipped Surrogate Objective)
        ratio = jnp.exp(log_probs - batch['old_log_probs'])
        surr1 = ratio * batch['advantages']
        surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch['advantages']
        actor_loss = -jnp.minimum(surr1, surr2).mean()
        
        # 5. PPO Critic Loss (Mean Squared Error dari TD-Returns)
        value_loss = 0.5 * jnp.square(values - batch['returns']).mean()
        
        # 6. Entropy Penalty (Mendorong eksplorasi / mencegah model terlalu cepat yakin)
        probs = jax.nn.softmax(masked_logits)
        # Sum(P * logP)
        entropy = -jnp.sum(probs * log_probs_all, axis=-1).mean()
        
        # 7. Total Loss Kombinasi
        total_loss = actor_loss + (val_coef * value_loss) - (ent_coef * entropy)
        
        # Kembalikan tuple (Loss untuk gradient, dan metrik untuk logistik/monitoring)
        return total_loss, (actor_loss, value_loss, entropy)
        
    # Hitung nilai Loss dan Gradient secara sekuensial
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_metrics), grads = grad_fn(params)
    
    # Hitung pembaruan optimasi menggunakan Optax (misal: Adam)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    
    # Aplikasikan pembaruan ke bobot (weights)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, aux_metrics

@partial(jax.jit, static_argnames=['apply_fn'])
def get_action_and_value(params, apply_fn, seq_input, glob_input, key):
    """
    Fungsi inferensi super-cepat untuk digunakan saat bermain (Rollout).
    """
    logits, values = apply_fn(params, seq_input, glob_input)
    
    # Action Masking
    action_mask = glob_input[..., 16:266]
    masked_logits = logits + ((1.0 - action_mask) * -1e9)
    
    # Distribusi probabilitas (Categorical)
    # Gunakan jax.random.categorical untuk sampling dari log_probs
    actions = jax.random.categorical(key, masked_logits, axis=-1)
    
    # Ambil log_prob dari aksi yang dipilih
    log_probs_all = jax.nn.log_softmax(masked_logits)
    log_probs = jnp.take_along_axis(log_probs_all, actions[..., None], axis=-1).squeeze(-1)
    
    return actions, log_probs, values.squeeze(-1), masked_logits
