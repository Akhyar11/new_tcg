import jax
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import flax.linen as nn

class PositionalEncoding(nn.Module):
    seq_len: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        # Menyuntikkan ID posisi ke urutan kartu
        pos_embedding = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.02),
            (1, self.seq_len, self.embed_dim)
        )
        return x + pos_embedding

class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int = 4
    ffn_dim: int = 512

    @nn.compact
    def __call__(self, x):
        # Menggunakan Pre-LayerNorm untuk stabilitas
        # Self-Attention + Residual
        norm_x = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim
        )(norm_x, norm_x)
        x = x + attn_out

        # FFN + Residual
        norm_x = nn.LayerNorm()(x)
        ffn_out = nn.Dense(self.ffn_dim)(norm_x)
        ffn_out = nn.swish(ffn_out)
        ffn_out = nn.Dense(self.embed_dim)(ffn_out)
        x = x + ffn_out

        return x

class CardEmbedding(nn.Module):
    vocab_size: int = 2000  # Estimasi maksimal unik card IDs
    embed_dim: int = 32

    @nn.compact
    def __call__(self, card_ids, tool_ids, pre_evo_ids, scalars):
        # card_ids, tool_ids, pre_evo_ids shape: (B, 93)
        # scalars shape: (B, 93, 28)
        
        card_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(card_ids)
        tool_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(tool_ids)
        pre_evo_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(pre_evo_ids)
        
        # Penjumlahan Additive
        total_emb = card_emb + tool_emb + pre_evo_emb # (B, 93, 32)
        
        # Penggabungan dengan scalar stats
        x = jnp.concatenate([total_emb, scalars], axis=-1) # (B, 93, 32 + 28) = (B, 93, 60)
        return x

class PokemonAgent(nn.Module):
    num_actions: int = 250
    embed_dim: int = 128

    @nn.compact
    def __call__(self, seq_input, glob_input):
        # 1. Card Embedding
        # seq_input shape: (B, 93, 31) -> 3 ID + 28 Skalar
        card_ids = seq_input[:, :, 0].astype(jnp.int32)
        tool_ids = seq_input[:, :, 1].astype(jnp.int32)
        pre_evo_ids = seq_input[:, :, 2].astype(jnp.int32)
        scalars = seq_input[:, :, 3:]
        
        x = CardEmbedding()(card_ids, tool_ids, pre_evo_ids, scalars) # (B, 93, 60)

        # 2. Sequence Processing
        # x shape: (B, 93, 60) -> Linear Projection (B, 93, 128)
        x = nn.Dense(self.embed_dim)(x)
        x = PositionalEncoding(seq_len=93, embed_dim=self.embed_dim)(x)
        
        # 3x Transformer Layers
        for _ in range(3):
            x = TransformerBlock(embed_dim=self.embed_dim)(x)

        # Slicing & Flattening
        my_hand = jnp.mean(x[:, 0:20, :], axis=1)            # (B, 128)
        my_discard = jnp.mean(x[:, 20:50, :], axis=1)        # (B, 128)
        opp_discard = jnp.mean(x[:, 50:80, :], axis=1)       # (B, 128)
        board_slots = x[:, 80:92, :].reshape(x.shape[0], -1) # (B, 1536) Flatten
        stadium_slot = x[:, 92, :]                           # (B, 128) Direct

        # 2b. Global Processing
        # glob_input shape: (B, 266)
        glob_x = nn.Dense(64)(glob_input)
        glob_x = nn.swish(glob_x)

        # 3. Main MLP Trunk (Fusion)
        fused = jnp.concatenate([
            my_hand, 
            my_discard, 
            opp_discard, 
            board_slots, 
            stadium_slot, 
            glob_x
        ], axis=-1)

        mlp_1 = nn.Dense(256)(fused)
        mlp_1 = nn.swish(mlp_1)
        mlp_1 = nn.LayerNorm()(mlp_1)

        mlp_2 = nn.Dense(256)(mlp_1)
        mlp_2 = nn.swish(mlp_2)

        # Add (Residual Connection)
        res_add = mlp_1 + mlp_2

        # 4. Output Heads
        # Action Masking Extraction from glob_input (Index 16-265)
        action_mask = glob_input[:, 16:16 + self.num_actions]

        # Actor Head (Policy)
        logits = nn.Dense(self.num_actions)(res_add)
        logits = jnp.clip(logits, -10.0, 10.0) # Mencegah logits meledak yang bisa menyebabkan NaN
        # Action Masking (logits - 1e9)
        masked_logits = jnp.where(action_mask == 1.0, logits, logits - 1e9)

        # Critic Head (Value)
        value = nn.Dense(1)(res_add)
        value = nn.tanh(value) # Bounds to [-1.0, 1.0]

        return masked_logits, value
