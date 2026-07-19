import jax
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import flax.linen as nn

class PositionalEncoding(nn.Module):
    seq_len: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
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
        norm_x = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim
        )(norm_x, norm_x)
        x = x + attn_out

        norm_x = nn.LayerNorm()(x)
        ffn_out = nn.Dense(self.ffn_dim)(norm_x)
        ffn_out = nn.swish(ffn_out)
        ffn_out = nn.Dense(self.embed_dim)(ffn_out)
        x = x + ffn_out

        return x

class CardEmbedding(nn.Module):
    vocab_size: int = 2000
    embed_dim: int = 32

    @nn.compact
    def __call__(self, card_ids, tool_ids, pre_evo_ids, scalars):
        shared_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim, name="knowledge_embed")
        
        card_emb = shared_embed(card_ids)
        tool_emb = shared_embed(tool_ids)
        pre_evo_emb = shared_embed(pre_evo_ids)
        
        card_emb = jax.lax.stop_gradient(card_emb)
        tool_emb = jax.lax.stop_gradient(tool_emb)
        pre_evo_emb = jax.lax.stop_gradient(pre_evo_emb)
        
        proj_card = nn.Dense(self.embed_dim, name="proj_card")(card_emb)
        proj_tool = nn.Dense(self.embed_dim, name="proj_tool")(tool_emb)
        proj_pre_evo = nn.Dense(self.embed_dim, name="proj_pre_evo")(pre_evo_emb)
        
        total_emb = proj_card + proj_tool + proj_pre_evo
        
        x = jnp.concatenate([total_emb, scalars], axis=-1)
        return x

class AttentionPooling(nn.Module):
    embed_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, mask=None):
        # 1. Definisikan token Query belajar khusus
        query = self.param(
            'pool_query', 
            nn.initializers.normal(stddev=0.02), 
            (1, 1, self.embed_dim)
        )
        query = jnp.tile(query, (x.shape[0], 1, 1))
        
        # 2. Siapkan mask untuk attention
        attn_mask = None
        if mask is not None:
            # Pastikan tipe boolean untuk mask attention
            attn_mask = (mask > 0.5)[:, jnp.newaxis, jnp.newaxis, :]
        
        # 3. Hitung Cross-Attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim
        )(query, x, mask=attn_mask)
        
        return jnp.squeeze(attn_out, axis=1)

class PokemonAgent(nn.Module):
    num_actions: int = 250
    embed_dim: int = 128

    @nn.compact
    def __call__(self, seq_input, glob_input, carry):
        # 1. Card Embedding
        card_ids = seq_input[:, :, 0].astype(jnp.int32)
        tool_ids = seq_input[:, :, 1].astype(jnp.int32)
        pre_evo_ids = seq_input[:, :, 2].astype(jnp.int32)
        scalars = seq_input[:, :, 3:]
        
        x = CardEmbedding()(card_ids, tool_ids, pre_evo_ids, scalars)

        # 2. Sequence Processing
        x = nn.Dense(self.embed_dim)(x)
        x = PositionalEncoding(seq_len=173, embed_dim=self.embed_dim)(x)
        
        for _ in range(3):
            x = TransformerBlock(embed_dim=self.embed_dim)(x)

        # Transformer bebas belajar dari sinyal RL.
        # Embedding (KB Distillation) tetap frozen via stop_gradient di CardEmbedding (line 54-56).

        # 3. Attention Pooling
        hand_pooler = AttentionPooling(embed_dim=self.embed_dim, name="hand_pooler")
        my_discard_pooler = AttentionPooling(embed_dim=self.embed_dim, name="my_discard_pooler")
        opp_discard_pooler = AttentionPooling(embed_dim=self.embed_dim, name="opp_discard_pooler")
        opp_known_hand_pooler = AttentionPooling(embed_dim=self.embed_dim, name="opp_known_hand_pooler")
        deck_pooler = AttentionPooling(embed_dim=self.embed_dim, name="deck_pooler")

        my_hand = hand_pooler(x[:, 0:20, :], mask=seq_input[:, 0:20, 15])
        my_discard = my_discard_pooler(x[:, 20:50, :], mask=seq_input[:, 20:50, 15])
        opp_discard = opp_discard_pooler(x[:, 50:80, :], mask=seq_input[:, 50:80, 15])
        board_slots = x[:, 80:92, :].reshape(x.shape[0], -1)
        stadium_slot = x[:, 92, :]
        opp_known_hand = opp_known_hand_pooler(x[:, 93:113, :], mask=seq_input[:, 93:113, 15])
        my_deck = deck_pooler(x[:, 113:173, :], mask=seq_input[:, 113:173, 15])

        # 2b. Global Processing
        glob_x = nn.Dense(64)(glob_input)
        glob_x = nn.swish(glob_x)

        # 3b. Main MLP Trunk (Fusion)
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

        res_add = mlp_1 + mlp_2
        
        # We don't freeze res_add because we froze x directly, which is more precise.
        new_carry, lstm_out = nn.LSTMCell(features=256)(carry, res_add)

        # 4. Output Heads
        # Actor Head (Pointer Network)
        # a. Proyeksikan state internal (lstm_out) menjadi Query taktik
        query = nn.Dense(self.embed_dim, name="action_query")(lstm_out) # (B, 128)
        
        # b. Definisikan static action keys parameter (250, 128)
        action_keys = self.param(
            'action_keys',
            nn.initializers.normal(stddev=0.02),
            (self.num_actions, self.embed_dim)
        )
        
        # c. Susun Key Matrix dinamis dengan menjumlahkan static action_keys dengan output embedding transformer 'x'
        keys = jnp.tile(action_keys[jnp.newaxis, :, :], (x.shape[0], 1, 1)) # (B, 250, 128)
        
        # PLAY (0-19) -> My Hand (0-19)
        keys = keys.at[:, 0:20, :].add(x[:, 0:20, :])
        
        # CARD (60-71) -> My/Opp Active & Bench
        keys = keys.at[:, 60, :].add(x[:, 86, :])       # Opp Active (86)
        keys = keys.at[:, 61:66, :].add(x[:, 87:92, :])   # Opp Bench (87-91)
        keys = keys.at[:, 66, :].add(x[:, 80, :])       # My Active (80)
        keys = keys.at[:, 67:72, :].add(x[:, 81:86, :])   # My Bench (81-85)
        
        # CARD (72-119) -> Deck Search cards (slots 113-160)
        keys = keys.at[:, 72:120, :].add(x[:, 113:161, :])
        
        # ATTACH (120-125) -> My Active (80) & My Bench (81-85)
        keys = keys.at[:, 120, :].add(x[:, 80, :])
        keys = keys.at[:, 121:126, :].add(x[:, 81:86, :])
        
        # EVOLVE (140-145) -> My Active (80) & My Bench (81-85)
        keys = keys.at[:, 140, :].add(x[:, 80, :])
        keys = keys.at[:, 141:146, :].add(x[:, 81:86, :])
        
        # RETREAT (161) -> My Active (80)
        keys = keys.at[:, 161, :].add(x[:, 80, :])
        
        # ATTACK (162-167) -> My Active (80)
        keys = keys.at[:, 162:168, :].add(x[:, 80, jnp.newaxis, :])
        
        # ABILITY (168-179) -> My/Opp Active & Bench
        keys = keys.at[:, 168, :].add(x[:, 80, :])
        keys = keys.at[:, 169:174, :].add(x[:, 81:86, :])
        keys = keys.at[:, 174, :].add(x[:, 86, :])
        keys = keys.at[:, 175:180, :].add(x[:, 87:92, :])
        
        # d. Dot product untuk logit policy
        logits = jnp.sum(query[:, jnp.newaxis, :] * keys, axis=-1)
        logits = jnp.clip(logits, -10.0, 10.0)

        # Critic Head (Value) - tetap sama
        value = nn.Dense(1, kernel_init=nn.initializers.normal(stddev=0.2))(lstm_out)
        value = jnp.tanh(value) * 5.0

        return logits, value, new_carry
