import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import flax.linen as nn

# Setup environment for CPU-only initialization
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

from tcg_core.models.ptr import PokemonAgent
from tcg_core.kaggle_sync import upload_to_kaggle

def main():
    print("=== Initialize Model v2 with Knowledge Distillation ===")
    
    ROOT = os.path.dirname(os.path.abspath(__file__))
    kd_path = os.path.join(ROOT, "knowledge_distillation", "student_embeddings_32d.npy")
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        
    print(f"Loading Knowledge Distillation embeddings from {kd_path}...")
    try:
        student_embeddings = np.load(kd_path)
        print(f"Loaded embeddings shape: {student_embeddings.shape}")
    except Exception as e:
        print(f"Error loading KD embeddings: {e}")
        return

    print("Initializing Pointer Network Model from scratch...")
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    
    # Dummy inputs for initialization
    dummy_seq = jnp.zeros((1, 173, 31))
    dummy_glob = jnp.zeros((1, 266))
    carry = nn.LSTMCell(features=256).initialize_carry(rng, (1,))
    
    params = model.init(rng, dummy_seq, dummy_glob, carry)
    
    # Convert frozen dict to mutable dict
    from flax.core import unfreeze, freeze
    mutable_params = unfreeze(params)
    
    # Inject Knowledge Distillation embeddings
    print("Injecting Knowledge Distillation embeddings into the model...")
    # Validate shapes
    expected_shape = mutable_params['params']['CardEmbedding_0']['knowledge_embed']['embedding'].shape
    if student_embeddings.shape[1] == expected_shape[1] and student_embeddings.shape[0] <= expected_shape[0]:
        num_embeddings = student_embeddings.shape[0]
        # Replace the first `num_embeddings` rows with KD embeddings
        mutable_params['params']['CardEmbedding_0']['knowledge_embed']['embedding'] = mutable_params['params']['CardEmbedding_0']['knowledge_embed']['embedding'].at[:num_embeddings, :].set(jnp.array(student_embeddings))
        print(f"Injection successful for {num_embeddings} embeddings.")
    else:
        print(f"Shape mismatch or dimension mismatch! Model expects {expected_shape}, but got {student_embeddings.shape}")
        return

    # Freeze params back
    final_params = freeze(mutable_params)
    
    # Save base and final weights
    base_path = os.path.join(checkpoints_dir, "model_lstm_pointer_v2_base.msgpack")
    final_path = os.path.join(checkpoints_dir, "model_lstm_pointer_v2_final.msgpack")
    
    print(f"Saving base weights to {base_path}...")
    with open(base_path, "wb") as f:
        f.write(serialization.to_bytes(final_params))
        
    print(f"Saving final weights to {final_path}...")
    with open(final_path, "wb") as f:
        f.write(serialization.to_bytes(final_params))
        
    print("\nUploading initialized models to Kaggle...")
    upload_to_kaggle(checkpoints_dir, message="Initial upload model_lstm_pointer_v2 with KD")
    print("\nDone! Model v2 is ready for training on Kaggle.")

if __name__ == "__main__":
    main()
