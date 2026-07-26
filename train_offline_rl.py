import os
import argparse
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import serialization
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from knowledge_distillation.dataset_loader import KaggleReplayDataset
from tcg_core.models.ptr import PokemonAgent as PTRModel

def create_train_state(rng, learning_rate):
    """Initializes the model and Optax TrainState."""
    model = PTRModel(num_actions=250)
    
    # Initialize variables with dummy inputs
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
def train_step(state, seq_batch, glob_batch, target_action_batch, target_value_batch, carry_batch, value_coef=0.5):
    """Perform a single training step using Cross Entropy Loss (Actor) and MSE Loss (Critic)."""
    def loss_fn(params):
        # Forward pass
        logits, values, new_carry = state.apply_fn(
            params, 
            seq_batch, 
            glob_batch, 
            carry_batch
        )
        
        # 1. Policy Loss (Actor) - Cross Entropy
        policy_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, 
            labels=target_action_batch
        )
        mean_policy_loss = jnp.mean(policy_loss)
        
        # 2. Value Loss (Critic) - Mean Squared Error
        # Ensure values output shape matches target shape
        values = jnp.squeeze(values, axis=-1)
        value_loss = jnp.square(values - target_value_batch)
        mean_value_loss = jnp.mean(value_loss)
        
        # Total Loss
        total_loss = mean_policy_loss + (value_coef * mean_value_loss)
        
        return total_loss, (mean_policy_loss, mean_value_loss, new_carry)

    (loss, (p_loss, v_loss, new_carry)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, p_loss, v_loss, new_carry

def main(args):
    dataset_dir = args.dataset_dir
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found.")
        return
        
    print(f"Loading dataset from {dataset_dir}...")
    dataset = KaggleReplayDataset(dataset_dir, max_files=args.max_files, gamma=args.gamma)
    
    if len(dataset) == 0:
        print("No valid winner samples found. Exiting.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, args.learning_rate)
    
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Loading weights from {args.load_checkpoint}...")
        with open(args.load_checkpoint, 'rb') as f:
            state = state.replace(params=serialization.from_bytes(state.params, f.read()))
            
    print("Starting Filtered Offline Actor-Critic training...")
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_p_loss = 0.0
        epoch_v_loss = 0.0
        batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for seq_input, glob_input, target_action, target_value in pbar:
            seq_jax = jnp.array(seq_input.numpy())
            glob_jax = jnp.array(glob_input.numpy())
            target_a_jax = jnp.array(target_action.numpy())
            target_v_jax = jnp.array(target_value.numpy())
            
            batch_size = seq_jax.shape[0]
            carry_jax = (jnp.zeros((batch_size, 256)), jnp.zeros((batch_size, 256)))
            
            state, total_loss, p_loss, v_loss, _ = train_step(
                state, seq_jax, glob_jax, target_a_jax, target_v_jax, carry_jax, args.value_coef
            )
            
            epoch_loss += total_loss.item()
            epoch_p_loss += p_loss.item()
            epoch_v_loss += v_loss.item()
            batches += 1
            pbar.set_postfix(loss=epoch_loss/batches, p_loss=epoch_p_loss/batches, v_loss=epoch_v_loss/batches)
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/batches:.4f} (Policy: {epoch_p_loss/batches:.4f}, Value: {epoch_v_loss/batches:.4f})")
        
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"model_offline_rl_epoch_{epoch+1}.msgpack")
            with open(save_path, 'wb') as f:
                f.write(serialization.to_bytes(state.params))
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PTRModel using Filtered Offline Actor-Critic")
    parser.add_argument("--dataset_dir", type=str, default="/kaggle/input/datasets/organizations/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-24", help="Path to Kaggle replay JSONs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--value_coef", type=float, default=0.5, help="Coefficient for Value (Critic) Loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for Value targets")
    parser.add_argument("--max_files", type=int, default=None, help="Max number of JSON files to process (for debugging)")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to existing .msgpack weights")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    main(args)
