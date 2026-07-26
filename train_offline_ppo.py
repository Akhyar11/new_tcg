import os
import argparse
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import serialization
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from knowledge_distillation.dataset_loader import KaggleReplayDataset
from tcg_core.models.ptr import PokemonAgent as PTRModel
from tcg_core.kaggle_sync import download_from_kaggle, upload_to_kaggle
import tcg_core.action_mapping as action_mapping

def collate_fn_filter_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def create_train_state(rng, learning_rate):
    model = PTRModel(num_actions=250)
    dummy_seq = jnp.zeros((1, 173, 31))
    dummy_glob = jnp.zeros((1, 266))
    dummy_carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
    
    variables = model.init(rng, dummy_seq, dummy_glob, dummy_carry)
    params = variables['params'] if 'params' in variables else variables
    
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

@jax.jit
def compute_old_log_probs_and_values_seq(state, seq_batch, glob_batch, carry_init, action_batch, mask_batch):
    """
    Melakukan unroll LSTM (lax.scan) untuk mencari log_prob lama dari seluruh sequence.
    Input dims: (Batch, Time, ...)
    """
    def scan_fn(carry, step_inputs):
        seq_t, glob_t, action_t, mask_t = step_inputs
        
        logits, values, new_carry = state.apply_fn({'params': state.params}, seq_t, glob_t, carry)
        values = jnp.squeeze(values, axis=-1)
        
        masked_logits = logits - 1e9 * (1.0 - mask_t)
        log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
        
        old_log_probs = jnp.take_along_axis(log_probs_all, jnp.expand_dims(action_t, axis=-1), axis=-1)
        old_log_probs = jnp.squeeze(old_log_probs, axis=-1)
        
        return new_carry, (old_log_probs, values)

    # Transpose input agar Time dimension ada di paling depan (Time, Batch, ...)
    seq_time_first = jnp.swapaxes(seq_batch, 0, 1)
    glob_time_first = jnp.swapaxes(glob_batch, 0, 1)
    action_time_first = jnp.swapaxes(action_batch, 0, 1)
    mask_time_first = jnp.swapaxes(mask_batch, 0, 1)
    
    _, (old_log_probs_t, old_values_t) = jax.lax.scan(
        scan_fn, 
        carry_init, 
        (seq_time_first, glob_time_first, action_time_first, mask_time_first)
    )
    
    # Kembalikan shape ke (Batch, Time, ...)
    old_log_probs = jnp.swapaxes(old_log_probs_t, 0, 1)
    old_values = jnp.swapaxes(old_values_t, 0, 1)
    
    return jax.lax.stop_gradient(old_log_probs), jax.lax.stop_gradient(old_values)

@jax.jit
def ppo_train_step_seq(state, seq_batch, glob_batch, carry_init, action_batch, mask_batch, 
                       target_value_batch, valid_mask_batch, old_log_probs, clip_eps=0.2, 
                       value_coef=0.5, entropy_coef=0.01):
    def loss_fn(params):
        def scan_fn(carry, step_inputs):
            seq_t, glob_t, action_t, mask_t, target_v_t, valid_t, old_log_prob_t = step_inputs
            
            logits, values, new_carry = state.apply_fn({'params': params}, seq_t, glob_t, carry)
            values = jnp.squeeze(values, axis=-1)
            
            masked_logits = logits - 1e9 * (1.0 - mask_t)
            log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
            probs_all = jax.nn.softmax(masked_logits, axis=-1)
            
            log_probs = jnp.take_along_axis(log_probs_all, jnp.expand_dims(action_t, axis=-1), axis=-1)
            log_probs = jnp.squeeze(log_probs, axis=-1)
            
            # GAE / Advantage (Monte Carlo simple estimation over time)
            advantages = target_v_t - jax.lax.stop_gradient(values)
            
            # Policy Loss
            ratio = jnp.exp(log_probs - old_log_prob_t)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -jnp.minimum(surr1, surr2)
            
            # Value Loss
            value_loss = jnp.square(values - target_v_t)
            
            # Entropy
            entropy = -jnp.sum(probs_all * log_probs_all, axis=-1)
            
            # Total Step Loss (Masked)
            step_loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy) * valid_t
            
            return new_carry, (step_loss, policy_loss * valid_t, value_loss * valid_t, entropy * valid_t)

        seq_time_first = jnp.swapaxes(seq_batch, 0, 1)
        glob_time_first = jnp.swapaxes(glob_batch, 0, 1)
        action_time_first = jnp.swapaxes(action_batch, 0, 1)
        mask_time_first = jnp.swapaxes(mask_batch, 0, 1)
        target_v_time_first = jnp.swapaxes(target_value_batch, 0, 1)
        valid_time_first = jnp.swapaxes(valid_mask_batch, 0, 1)
        old_log_probs_time_first = jnp.swapaxes(old_log_probs, 0, 1)
        
        _, (step_losses, p_losses, v_losses, e_losses) = jax.lax.scan(
            scan_fn, 
            carry_init, 
            (seq_time_first, glob_time_first, action_time_first, mask_time_first, 
             target_v_time_first, valid_time_first, old_log_probs_time_first)
        )
        
        valid_sum = jnp.sum(valid_mask_batch) + 1e-8
        
        # Mean across batch and time (only for valid steps)
        mean_total_loss = jnp.sum(step_losses) / valid_sum
        mean_p_loss = jnp.sum(p_losses) / valid_sum
        mean_v_loss = jnp.sum(v_losses) / valid_sum
        mean_e_loss = jnp.sum(e_losses) / valid_sum
        
        return mean_total_loss, (mean_p_loss, mean_v_loss, mean_e_loss)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux[0], aux[1], aux[2]

def main(args):
    dataset_dir = args.dataset_dir
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found.")
        return
        
    print(f"Loading dataset dari {dataset_dir} (Max Steps: {args.max_steps})...")
    dataset = KaggleReplayDataset(dataset_dir, max_files=args.max_files, gamma=args.gamma, max_steps=args.max_steps)
    
    if len(dataset) == 0:
        print("No valid sequences found. Exiting.")
        return
        
    print(f"Syncing initial weights from Kaggle...")
    download_from_kaggle(args.save_dir)
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=4,
        collate_fn=collate_fn_filter_none
    )
    
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, args.learning_rate)
    
    model_path = os.path.join(args.save_dir, "model_ptr_v3.msgpack")
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        with open(model_path, 'rb') as f:
            state = state.replace(params=serialization.from_bytes(state.params, f.read()))
    elif args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Loading fallback weights from {args.load_checkpoint}...")
        with open(args.load_checkpoint, 'rb') as f:
            state = state.replace(params=serialization.from_bytes(state.params, f.read()))
            
    print("Starting Sequence-Based Offline PPO training...")
    
    best_loss = float('inf')
    
    try:
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            epoch_p_loss = 0.0
            epoch_v_loss = 0.0
            epoch_ent_loss = 0.0
            batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch_data in pbar:
                if batch_data is None:
                    continue
                seq_input, glob_input, target_action, target_value, action_mask, valid_mask = batch_data
                seq_jax = jnp.array(seq_input.numpy())
                glob_jax = jnp.array(glob_input.numpy())
                target_a_jax = jnp.array(target_action.numpy())
                target_v_jax = jnp.array(target_value.numpy())
                mask_jax = jnp.array(action_mask.numpy())
                valid_jax = jnp.array(valid_mask.numpy())
                
                batch_size = seq_jax.shape[0]
                # Carry state selalu direset ke 0 di awal setiap ronde/episode game
                carry_init = (jnp.zeros((batch_size, 256)), jnp.zeros((batch_size, 256)))
                
                # Step 1: Hitung old_log_probs secara sequence (unrolled over time)
                old_log_probs, _ = compute_old_log_probs_and_values_seq(
                    state, seq_jax, glob_jax, carry_init, target_a_jax, mask_jax
                )
                
                # Normalisasi keuntungan di level batch untuk kestabilan pelatihan
                # Ini dilakukan di dalam lax.scan di atas secara implisit atau bisa dipindahkan ke sini jika mau lebih presisi.
                
                # Step 2: Lakukan PPO Update di sepanjang sequence
                for _ in range(args.ppo_epochs):
                    state, total_loss, p_loss, v_loss, ent_loss = ppo_train_step_seq(
                        state, seq_jax, glob_jax, carry_init, target_a_jax, mask_jax, 
                        target_v_jax, valid_jax, old_log_probs, 
                        clip_eps=args.clip_eps, 
                        value_coef=args.value_coef, 
                        entropy_coef=args.entropy_coef
                    )
                
                epoch_loss += total_loss.item()
                epoch_p_loss += p_loss.item()
                epoch_v_loss += v_loss.item()
                epoch_ent_loss += ent_loss.item()
                batches += 1
                pbar.set_postfix(
                    L=f"{epoch_loss/batches:.3f}", 
                    P=f"{epoch_p_loss/batches:.3f}", 
                    V=f"{epoch_v_loss/batches:.3f}"
                )
                
            avg_epoch_loss = epoch_loss / batches
            print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f} (Pol: {epoch_p_loss/batches:.4f}, Val: {epoch_v_loss/batches:.4f}, Ent: {epoch_ent_loss/batches:.4f})")
            
            if avg_epoch_loss < best_loss:
                print(f"Loss improved from {best_loss:.4f} to {avg_epoch_loss:.4f}. Saving best model...")
                best_loss = avg_epoch_loss
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    with open(model_path, 'wb') as f:
                        f.write(serialization.to_bytes(state.params))
                    print(f"Saved best checkpoint to {model_path}")
            else:
                print(f"Loss did not improve from {best_loss:.4f}.")
                
        print("\nUploading final best model to Kaggle...")
        upload_to_kaggle(args.save_dir, message="Sync best model_ptr_v3 from Offline PPO")
        print("Done.")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saved current progress.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PTRModel using Sequence-Based Offline PPO")
    parser.add_argument("--dataset_dir", type=str, default="/kaggle/input/datasets/organizations/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-24", help="Path to Kaggle replay JSONs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=256, help="Panjang maksimal sekuens (time dimension)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ppo_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    main(args)
