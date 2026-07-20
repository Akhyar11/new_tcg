import jax
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import optax
from functools import partial

@partial(jax.pmap, in_axes=(0, 0, 0, None, None, None, None), static_broadcasted_argnums=(3, 4), axis_name='gpu')
def ppo_update_step(params, opt_state, batch, apply_fn, tx, clip_ratio, entropy_coef=0.01):
    """
    Satu langkah optimasi (Gradient Descent) menggunakan TBPTT.
    """

    def loss_fn(p):
        # 1. Forward Pass melalui LSTM via lax.scan
        # init_carry: (B, 256)
        init_carry = (batch['carry_c'], batch['carry_h'])
        
        # scan membutuhkan iterasi pada dimensi 0 (waktu). 
        # Saat ini batch berdimensi (B, seq_len, ...). Swap ke (seq_len, B, ...)
        def swap_batch_seq(x):
            return jnp.swapaxes(x, 0, 1)
            
        # Hanya ambil data yang diperlukan oleh model untuk scan (buang carry awal dsb)
        scan_inputs = {
            'seq_input': swap_batch_seq(batch['seq_input']),
            'glob_input': swap_batch_seq(batch['glob_input'])
        }

        def scan_step(carry, step_batch):
            logits, values, new_carry = apply_fn(p, step_batch['seq_input'], step_batch['glob_input'], carry)
            return new_carry, (logits, values)

        _, (all_logits, all_values) = jax.lax.scan(scan_step, init_carry, scan_inputs)
        
        # Kembalikan ke shape awal (B, seq_len, ...)
        logits = jnp.swapaxes(all_logits, 0, 1)
        values = jnp.swapaxes(all_values, 0, 1).squeeze(-1) # (B, seq_len)

        # 2. Action Masking
        action_mask = batch['glob_input'][..., 16:266]
        masked_logits = logits + ((1.0 - action_mask) * -1e9)

        # 3. Hitung Log Probabilities
        log_probs_all = jax.nn.log_softmax(masked_logits)

        log_probs = jnp.sum(log_probs_all * batch['actions_mask'], axis=-1)

        # 4. PPO Actor Loss
        log_ratio = jnp.clip(log_probs - batch['old_log_probs'], -10.0, 10.0)
        ratio = jnp.exp(log_ratio)

        active_mask = (batch['active_players'] == 0).astype(jnp.float32)
        valid_count = jnp.maximum(1.0, active_mask.sum())

        surr1 = ratio * batch['advantages'] * active_mask
        surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch['advantages'] * active_mask
        actor_loss = -jnp.minimum(surr1, surr2).sum() / valid_count

        # 5. PPO Critic Loss
        value_loss = 0.5 * (jnp.square(values - batch['returns']) * active_mask).sum() / valid_count

        # 6. Entropy Bonus
        probs = jax.nn.softmax(masked_logits)
        entropy = -jnp.sum(probs * log_probs_all * action_mask, axis=-1)
        entropy = (entropy * active_mask).sum() / valid_count

        # 7. Total Loss Kombinasi
        total_loss = actor_loss + (0.5 * value_loss) - (entropy_coef * entropy)

        return total_loss, (actor_loss, value_loss, entropy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_metrics), grads = grad_fn(params)

    grads = jax.lax.pmean(grads, axis_name='gpu')
    loss = jax.lax.pmean(loss, axis_name='gpu')

    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss, aux_metrics


@partial(jax.pmap, static_broadcasted_argnums=(1,), axis_name='gpu')
def get_action_and_value(params, apply_fn, seq_input, glob_input, carry, key):
    """
    Fungsi inferensi (Rollout). Menerima state saat ini (c, h).
    """
    logits, values, new_carry = apply_fn(params, seq_input, glob_input, carry)

    action_mask = glob_input[..., 16:266]
    masked_logits = logits + ((1.0 - action_mask) * -1e9)

    actions = jax.random.categorical(key, masked_logits, axis=-1)

    log_probs_all = jax.nn.log_softmax(masked_logits)
    log_probs = jnp.take_along_axis(log_probs_all, actions[..., None], axis=-1).squeeze(-1)

    return actions, log_probs, values.squeeze(-1), logits, new_carry
