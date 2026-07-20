import jax
import jax.numpy as jnp
import optax
import numpy as np
import time

from tcg_core.models.ff import PokemonAgent
from tcg_core.ppo_update import ppo_update_step, get_action_and_value

def test_ppo():
    print("1. Menginisiasi Model JAX dan Optimizer...")
    rng = jax.random.PRNGKey(0)
    model = PokemonAgent(num_actions=250)
    
    # Dummy data
    batch_size = 8
    seq_in = jnp.zeros((batch_size, 93, 31))
    glob_in = jnp.zeros((batch_size, 266))
    
    # Action mask (seolah-olah aksi 0 dan 1 saja yang legal)
    # Action mask ada di indeks 16 sampai 265
    glob_in = glob_in.at[:, 16:18].set(1.0)
    
    # Inisialisasi bobot
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, seq_in, glob_in)
    
    # Inisialisasi Optimizer
    tx = optax.adam(learning_rate=3e-4)
    opt_state = tx.init(params)
    
    print("2. Menguji Kompilasi JIT get_action_and_value...")
    start_time = time.time()
    rng, step_rng = jax.random.split(rng)
    # JIT Compilation terjadi di pemanggilan pertama
    actions, log_probs, values = get_action_and_value(params, model.apply, seq_in, glob_in, step_rng)
    actions.block_until_ready()
    print(f"   Kompilasi awal memakan waktu: {time.time() - start_time:.4f} detik")
    
    start_time = time.time()
    rng, step_rng = jax.random.split(rng)
    actions, log_probs, values = get_action_and_value(params, model.apply, seq_in, glob_in, step_rng)
    actions.block_until_ready()
    print(f"   Eksekusi ke-2 (setelah JIT): {time.time() - start_time:.6f} detik! Sangat cepat.")
    print(f"   Pilihan Aksi (Harus 0 atau 1 karena mask): {actions}")
    
    print("\n3. Menguji Kompilasi JIT ppo_update_step...")
    # Siapkan dummy batch
    batch = {
        "seq_input": seq_in,
        "glob_input": glob_in,
        "actions": actions,
        "old_log_probs": log_probs,
        "advantages": jnp.ones(batch_size),
        "returns": jnp.ones(batch_size) * 1.5,
    }
    
    start_time = time.time()
    new_params, new_opt_state, loss, aux = ppo_update_step(params, opt_state, batch, model.apply, tx)
    loss.block_until_ready()
    print(f"   Kompilasi awal Update memakan waktu: {time.time() - start_time:.4f} detik")
    
    start_time = time.time()
    new_params, new_opt_state, loss, aux = ppo_update_step(new_params, new_opt_state, batch, model.apply, tx)
    loss.block_until_ready()
    print(f"   Eksekusi Update ke-2 (setelah JIT): {time.time() - start_time:.6f} detik!")
    print(f"   Total Loss: {loss:.4f} | Actor Loss: {aux[0]:.4f} | Value Loss: {aux[1]:.4f}")

    print("\n[SEMUA PENGUJIAN JAX PPO BERHASIL LULUS!]")

if __name__ == "__main__":
    test_ppo()
