import os
import argparse
import jax
import jax.numpy as jnp
import functools
from flax.jax_utils import replicate, unreplicate
import optax
from flax.training import train_state
from flax import serialization
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from knowledge_distillation.dataset_loader import KaggleReplayDataset
from tcg_core.models.critic import CriticModel
from tcg_core.kaggle_sync import download_from_kaggle, upload_to_kaggle
import gc
import tcg_core.action_mapping as action_mapping

def collate_fn_filter_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def create_train_state(rng, learning_rate):
    model = CriticModel()
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

@functools.partial(jax.pmap, axis_name='batch')
def train_step_seq(state, seq_batch, glob_batch, carry_init, target_value_batch, valid_mask_batch):
    def loss_fn(params):
        @jax.remat
        def scan_fn(carry, step_inputs):
            seq_t, glob_t, target_v_t, valid_t = step_inputs
            
            value, new_carry = state.apply_fn({'params': params}, seq_t, glob_t, carry)
            value = jnp.squeeze(value, axis=-1)
            
            # Value Loss (MSE)
            value_loss = jnp.square(value - target_v_t)
            
            # Total Step Loss (Masked)
            step_loss = value_loss * valid_t
            
            return new_carry, step_loss

        seq_time_first = jnp.swapaxes(seq_batch, 0, 1)
        glob_time_first = jnp.swapaxes(glob_batch, 0, 1)
        target_v_time_first = jnp.swapaxes(target_value_batch, 0, 1)
        valid_time_first = jnp.swapaxes(valid_mask_batch, 0, 1)
        
        _, step_losses = jax.lax.scan(
            scan_fn, 
            carry_init, 
            (seq_time_first, glob_time_first, target_v_time_first, valid_time_first)
        )
        
        valid_sum = jnp.sum(valid_mask_batch) + 1e-8
        
        # Mean across batch and time (only for valid steps)
        mean_loss = jnp.sum(step_losses) / valid_sum
        return mean_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def main(args):
    dataset = KaggleReplayDataset(args.dataset_dir, max_steps=args.max_steps, for_rl=True)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_filter_none,
        num_workers=4,
        prefetch_factor=1
    )
    
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, args.learning_rate)
    
    num_devices = jax.local_device_count()
    print(f"Menggunakan {num_devices} GPU(s) via jax.pmap")
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model_path = os.path.join(args.save_dir, "model_critic_v1.msgpack")
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}...")
            with open(model_path, 'rb') as f:
                state = state.replace(params=serialization.from_bytes(state.params, f.read()))
        else:
            print("Syncing initial weights from Kaggle (if any)...")
            download_from_kaggle(args.save_dir)
            if os.path.exists(model_path):
                print(f"Loading weights from {model_path}...")
                with open(model_path, 'rb') as f:
                    state = state.replace(params=serialization.from_bytes(state.params, f.read()))
            else:
                base_path = os.path.join(args.save_dir, "model_lstm_pointer_final.msgpack")
                if os.path.exists(base_path):
                    print(f"Loading Teacher Weights from {base_path} (PTR V1)...")
                    with open(base_path, 'rb') as f:
                        base_params = serialization.from_bytes(None, f.read())
                        
                        from flax.core import unfreeze, freeze
                        unfrozen_params = unfreeze(state.params)
                        
                        # Copy semua layer/bobot yang memiliki nama sama (Transformer, LSTM, Embedding)
                        copied_keys = 0
                        for key in base_params.keys():
                            if key in unfrozen_params:
                                unfrozen_params[key] = base_params[key]
                                copied_keys += 1
                                
                        state = state.replace(params=freeze(unfrozen_params))
                        print(f"Berhasil memasang {copied_keys} layer dari otak Guru (PTR V1) ke Critic!")
                    
    # Replicate state to all devices AFTER loading weights
    state = replicate(state)
    
    def shard(x):
        return jnp.array(x).reshape(num_devices, -1, *x.shape[1:])
        
    print("Starting Offline Critic Training...")
    
    best_loss = float('inf')
    
    try:
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            batches = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                if batch is None:
                    continue
                    
                seq_input = batch['seq_inputs']
                glob_input = batch['glob_inputs']
                target_value = batch['target_values']
                valid_mask = batch['valid_masks']
                
                # Reshape to (num_devices, local_batch_size, ...)
                bs = seq_input.shape[0]
                if bs % num_devices != 0:
                    continue
                
                seq_jax = shard(seq_input.numpy())
                glob_jax = shard(glob_input.numpy())
                target_v_jax = shard(target_value.numpy())
                valid_jax = shard(valid_mask.numpy())
                
                carry_init_jax = (jnp.zeros((num_devices, bs // num_devices, 256)), 
                                  jnp.zeros((num_devices, bs // num_devices, 256)))
                
                state, loss = train_step_seq(
                    state, seq_jax, glob_jax, carry_init_jax, 
                    target_v_jax, valid_jax
                )
                
                loss_val = float(jnp.mean(loss))
                epoch_loss += loss_val
                batches += 1
                
                pbar.set_postfix(L=f"{loss_val:.3f}")
                
                # Cleanup memory
                del seq_input, glob_input, target_value, valid_mask
                del seq_jax, glob_jax, target_v_jax, valid_jax
                if batches % 10 == 0:
                    gc.collect()
                
            avg_epoch_loss = epoch_loss / max(1, batches)
            print(f"Epoch {epoch+1} Loss (MSE): {avg_epoch_loss:.4f}")
            
            if avg_epoch_loss < best_loss:
                print(f"Loss improved from {best_loss:.4f} to {avg_epoch_loss:.4f}. Saving best model...")
                best_loss = avg_epoch_loss
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    with open(model_path, 'wb') as f:
                        from flax.jax_utils import unreplicate
                        saved_state = unreplicate(state)
                        f.write(serialization.to_bytes(saved_state.params))
                    print(f"Saved best checkpoint to {model_path}")
                    upload_to_kaggle(args.save_dir, message=f"Sync best critic model from Epoch {epoch+1}")
            else:
                print(f"Loss did not improve from {best_loss:.4f}.")
                
        print("\nUploading final best model to Kaggle...")
        upload_to_kaggle(args.save_dir, message="Sync best critic model")
        print("Done.")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saved current progress.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CriticModel (Value Shaping)")
    parser.add_argument("--dataset_dir", type=str, default="/kaggle/input/datasets/organizations/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-24", help="Path to Kaggle replay JSONs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=256, help="Panjang maksimal sekuens (time dimension)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    main(args)
