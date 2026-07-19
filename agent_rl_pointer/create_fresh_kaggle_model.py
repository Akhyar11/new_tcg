import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.core import unfreeze, freeze

# Memastikan import dari root project berjalan lancar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_rl_pointer.model import PokemonAgent
from agent_rl_pointer.train import upload_to_kaggle, SAVE_DIR, root_dir

def generate_and_upload_base_model():
    print("=== GENERATE NEW BASE MODEL DENGAN 32D DISTILLED EMBEDDING ===")
    
    # 1. Inisialisasi arsitektur model RL yang baru
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42) # Seed fixed untuk reproducibility
    
    dummy_seq = jnp.zeros((1, 173, 31))
    dummy_glob = jnp.zeros((1, 266))
    dummy_carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
    
    print("1. Menginisialisasi parameter model random (scratch)...")
    params = model.init(rng, dummy_seq, dummy_glob, dummy_carry)
    
    # 1.5 Load bobot matching dari model_final.msgpack (FeedForward)
    model_final_path = os.path.join(SAVE_DIR, "model_final.msgpack")
    if os.path.exists(model_final_path):
        print(f"   -> Membaca bobot model_final FF dari {model_final_path}...")
        with open(model_final_path, 'rb') as f:
            bytes_data = f.read()
            try:
                old_params = serialization.msgpack_restore(bytes_data)
                
                def update_dict(target_d, old_d):
                    for k, v in old_d.items():
                        if k in target_d:
                            if isinstance(v, dict):
                                update_dict(target_d[k], v)
                            else:
                                if getattr(target_d[k], "shape", None) == v.shape:
                                    target_d[k] = v
                    return target_d
                
                params_mut = unfreeze(params)
                params_mut = update_dict(params_mut, old_params)
                params = freeze(params_mut)
                print("   -> Sukses menyalin bobot matching dari model_final FF!")
            except Exception as e:
                print(f"[!] Gagal menyalin bobot model_final FF: {e}")
    else:
        print(f"[!] PERINGATAN: model_final.msgpack tidak ditemukan di {model_final_path}. Memulai murni dari random.")

    # 2. Inject bobot Knowledge Distillation
    distill_path = os.path.join(root_dir, "knowledge_distillation", "student_embeddings_32d.npy")
    if os.path.exists(distill_path):
        print(f"2. Membaca bobot 32D dari {distill_path}...")
        knowledge_weights = np.load(distill_path)
        
        vocab_size = params['params']['CardEmbedding_0']['knowledge_embed']['embedding'].shape[0]
        embed_dim = params['params']['CardEmbedding_0']['knowledge_embed']['embedding'].shape[1]
        
        padded_weights = np.zeros((vocab_size, embed_dim))
        num_cards = min(knowledge_weights.shape[0], vocab_size)
        padded_weights[:num_cards, :] = knowledge_weights[:num_cards, :]
        
        # Proses Modifikasi JAX FrozenDict
        params_mut = unfreeze(params)
        params_mut['params']['CardEmbedding_0']['knowledge_embed']['embedding'] = jnp.array(padded_weights)
        params = freeze(params_mut)
        print("   -> Bobot 32D berhasil di-inject ke arsitektur model baru!")
    else:
        print("[!] PERINGATAN: File bobot student_embeddings_32d.npy tidak ditemukan. Model akan murni random.")

    # 3. Simpan parameter ke disk
    os.makedirs(SAVE_DIR, exist_ok=True)
    base_path = os.path.join(SAVE_DIR, "model_lstm_pointer_base.msgpack")
    final_path = os.path.join(SAVE_DIR, "model_lstm_pointer_final.msgpack")
    
    print(f"3. Menyimpan checkpoint ke:")
    print(f"   - {base_path}")
    with open(base_path, 'wb') as f:
        f.write(serialization.to_bytes(params))
        
    print(f"   - {final_path}")
    with open(final_path, 'wb') as f:
        f.write(serialization.to_bytes(params))

    # 4. Upload ke Kaggle
    print("4. Memulai upload dataset ke Kaggle...")
    try:
        upload_to_kaggle(SAVE_DIR, message="Fresh base model with 32D Knowledge Distillation Embeddings")
        print("=== PROSES SELESAI ===")
    except Exception as e:
        print(f"[!] Gagal upload ke Kaggle: {e}")

if __name__ == "__main__":
    generate_and_upload_base_model()
