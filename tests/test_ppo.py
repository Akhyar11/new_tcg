import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
import os
import sys

# Add ROOT to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from tcg_core.models.lstm import PokemonAgent
from tcg_core.ppo_update import ppo_update_step, get_action_and_value

def test_ppo():
    print("1. Menginisiasi Model JAX dan Optimizer untuk LSTM...")
    rng = jax.random.PRNGKey(0)
    model = PokemonAgent()
    
    # Dummy data
    # (devices, batch_per_device, seq_len, ...) -> CPU uses 1 device
    batch_size = 2
    seq_len = 32
    seq_in = jnp.zeros((1, batch_size, seq_len, 173, 31))
    glob_in = jnp.zeros((1, batch_size, seq_len, 266))
    
    # Action mask (seolah-olah aksi 0 dan 1 saja yang legal)
    # Action mask ada di indeks 16 sampai 265
    glob_in = glob_in.at[:, :, :, 16:18].set(1.0)
    
    # Carry dummy
    carry_c = jnp.zeros((1, batch_size, 256))
    carry_h = jnp.zeros((1, batch_size, 256))
    carry = (carry_c, carry_h)
    
    # Inisialisasi bobot (butuh seq_len=1 untuk inisialisasi)
    rng, init_rng = jax.random.split(rng)
    dummy_init_seq = jnp.zeros((1, 173, 31))
    dummy_init_glob = jnp.zeros((1, 266))
    dummy_init_carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
    
    from flax.jax_utils import replicate
    
    params = model.init(init_rng, dummy_init_seq, dummy_init_glob, dummy_init_carry)
    
    # Inisialisasi Optimizer sebelum replicate
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(learning_rate=3e-4)
    )
    opt_state = tx.init(params)
    
    # Replicate for pmap
    params = replicate(params)
    opt_state = replicate(opt_state)
    
    print("2. Menguji Kompilasi JIT get_action_and_value...")
    start_time = time.time()
    rng, step_rng = jax.random.split(rng)
    step_rngs = jax.random.split(step_rng, 1) # shape (1,2)
    
    # get_action_and_value runs on a single timestep, so we take seq_in[:, :, 0]
    single_seq = seq_in[:, :, 0, :, :]
    single_glob = glob_in[:, :, 0, :]
    
    # JIT Compilation terjadi di pemanggilan pertama
    actions, log_probs, values, logits, next_carry = get_action_and_value(params, model.apply, single_seq, single_glob, carry, step_rngs)
    actions.block_until_ready()
    print(f"   Kompilasi awal memakan waktu: {time.time() - start_time:.4f} detik")
    
    start_time = time.time()
    rng, step_rng = jax.random.split(rng)
    step_rngs = jax.random.split(step_rng, 1) # shape (1,2)
    actions, log_probs, values, logits, next_carry = get_action_and_value(params, model.apply, single_seq, single_glob, carry, step_rngs)
    actions.block_until_ready()
    print(f"   Eksekusi ke-2 (setelah JIT): {time.time() - start_time:.6f} detik! Sangat cepat.")
    
    print("\n3. Menguji Kompilasi JIT ppo_update_step...")
    # Siapkan dummy batch
    batch = {
        "seq_input": seq_in,
        "glob_input": glob_in,
        "actions": jnp.zeros((1, batch_size, seq_len), dtype=jnp.int32),
        "old_log_probs": jnp.zeros((1, batch_size, seq_len)),
        "advantages": jnp.ones((1, batch_size, seq_len)),
        "returns": jnp.ones((1, batch_size, seq_len)) * 1.5,
        "carry_c": carry_c,
        "carry_h": carry_h,
        "actions_mask": jnp.ones((1, batch_size, seq_len, 250)),
        "active_players": jnp.zeros((1, batch_size, seq_len))
    }
    
    start_time = time.time()
    new_params, new_opt_state, loss, aux = ppo_update_step(params, opt_state, batch, model.apply, tx, 0.2, 0.01)
    loss.block_until_ready()
    print(f"   Kompilasi awal Update memakan waktu: {time.time() - start_time:.4f} detik")
    
    start_time = time.time()
    new_params, new_opt_state, loss, aux = ppo_update_step(new_params, new_opt_state, batch, model.apply, tx, 0.2, 0.01)
    loss.block_until_ready()
    print(f"   Eksekusi Update ke-2 (setelah JIT): {time.time() - start_time:.6f} detik!")
    print(f"   Total Loss: {loss[0]:.4f} | Actor Loss: {aux[0][0]:.4f} | Value Loss: {aux[1][0]:.4f}")

    print("\n[SEMUA PENGUJIAN JAX PPO BERHASIL LULUS!]")

if __name__ == "__main__":
    test_ppo()
