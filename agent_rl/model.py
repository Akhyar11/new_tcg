import flax.linen as nn
import jax.numpy as jnp

class PokemonAgent(nn.Module):
    num_actions: int = 200

    @nn.compact
    def __call__(self, features, action_mask):
        # Flatten representasi board
        board_flat = features["board"].reshape((features["board"].shape[0], -1))
        
        # Gabungkan semua fitur menjadi 1 vektor
        x = jnp.concatenate([
            features["global"], 
            board_flat[0],  # Board kita
            board_flat[1],  # Board musuh
            features["hand"]
        ], axis=-1)
        
        # Jaringan Syaraf (MLP)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # Output logits untuk masing-masing kemungkinan aksi
        logits = nn.Dense(self.num_actions)(x)
        
        # ACTION MASKING: Ubah probabilitas opsi yang tidak valid jadi hampir nol
        masked_logits = jnp.where(action_mask, logits, -1e9)
        
        return masked_logits
