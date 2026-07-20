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
        # card_ids, tool_ids, pre_evo_ids shape: (B, 173)
        # scalars shape: (B, 173, 28)
        
        # 1. Gunakan SATU layer embedding agar bobotnya sama (shared weights)
        shared_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim, name="knowledge_embed")
        
        card_emb = shared_embed(card_ids)
        tool_emb = shared_embed(tool_ids)
        pre_evo_emb = shared_embed(pre_evo_ids)
        
        # 2. BEKUKAN (FREEZE) layer embedding! 
        # JAX akan memotong gradien di titik ini sehingga bobot dari Xenova tidak rusak saat training RL.
        card_emb = jax.lax.stop_gradient(card_emb)
        tool_emb = jax.lax.stop_gradient(tool_emb)
        pre_evo_emb = jax.lax.stop_gradient(pre_evo_emb)
        
        # 3. Proyeksikan masing-masing secara linear ke dimensi embed_dim sebelum dijumlahkan.
        # Ini memberikan fleksibilitas bagi model RL untuk mengadaptasi bobot semantik yang dibekukan.
        proj_card = nn.Dense(self.embed_dim, name="proj_card")(card_emb)
        proj_tool = nn.Dense(self.embed_dim, name="proj_tool")(tool_emb)
        proj_pre_evo = nn.Dense(self.embed_dim, name="proj_pre_evo")(pre_evo_emb)
        
        total_emb = proj_card + proj_tool + proj_pre_evo # (B, 173, 32)
        
        # Penggabungan dengan scalar stats
        x = jnp.concatenate([total_emb, scalars], axis=-1) # (B, 173, 32 + 28) = (B, 173, 60)
        return x

class PokemonAgent(nn.Module):
    num_actions: int = 250
    embed_dim: int = 128

    @nn.compact
    def __call__(self, seq_input, glob_input):
        # 1. Card Embedding
        # seq_input shape: (B, 173, 31) -> 3 ID + 28 Skalar
        card_ids = seq_input[:, :, 0].astype(jnp.int32)
        tool_ids = seq_input[:, :, 1].astype(jnp.int32)
        pre_evo_ids = seq_input[:, :, 2].astype(jnp.int32)
        scalars = seq_input[:, :, 3:]
        
        x = CardEmbedding()(card_ids, tool_ids, pre_evo_ids, scalars) # (B, 173, 60)

        # 2. Sequence Processing
        # x shape: (B, 173, 60) -> Linear Projection (B, 173, 128)
        x = nn.Dense(self.embed_dim)(x)
        x = PositionalEncoding(seq_len=173, embed_dim=self.embed_dim)(x)
        
        # 3x Transformer Layers
        for _ in range(3):
            x = TransformerBlock(embed_dim=self.embed_dim)(x)

        # Slicing & Flattening
        my_hand = jnp.mean(x[:, 0:20, :], axis=1)            # (B, 128)
        my_discard = jnp.mean(x[:, 20:50, :], axis=1)        # (B, 128)
        opp_discard = jnp.mean(x[:, 50:80, :], axis=1)       # (B, 128)
        board_slots = x[:, 80:92, :].reshape(x.shape[0], -1) # (B, 1536) Flatten
        stadium_slot = x[:, 92, :]                           # (B, 128) Direct
        opp_known_hand = jnp.mean(x[:, 93:113, :], axis=1)   # (B, 128)
        my_deck = jnp.mean(x[:, 113:173, :], axis=1)         # (B, 128)

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
            opp_known_hand,
            my_deck,
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

        # Actor Head (Policy) — return raw logits. Masking dilakukan di PPO update.
        logits = nn.Dense(self.num_actions)(res_add)
        logits = jnp.clip(logits, -10.0, 10.0) # Mencegah logits meledak yang bisa menyebabkan NaN

        # Critic Head (Value) — tanh bounded ke [-5, +5]
        # kernel_init std=0.2: cukup besar agar value bisa belajar dari awal
        # (tidak stuck di ~0 saat returns=±3), cukup kecil untuk gradien stabil.
        # Tanh bounding mencegah nilai absurd yang merusak GAE.
        value = nn.Dense(1, kernel_init=nn.initializers.normal(stddev=0.2))(res_add)
        value = jnp.tanh(value) * 5.0  # Bound: [-5, +5]

        return logits, value
