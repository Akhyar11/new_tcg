import os
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import flax.linen as nn
import optax
import numpy as np
from tqdm import tqdm

class StudentEmbedding(nn.Module):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(x)

# Fungsi untuk menghitung Cosine Similarity secara batch di JAX
def cosine_similarity(u, v):
    u_norm = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
    v_norm = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.sum(u_norm * v_norm, axis=-1)

def train_student():
    csv_path = "card_similarity_scores.csv"
    output_weights = "student_embeddings_32d.npy"
    embedding_dim = 32  # Menggunakan 32 dimensi sesuai permintaan!
    
    print("Membaca dataset kemiripan (Teacher Targets)...")
    df = pd.read_csv(csv_path)
    
    # Ambil nilai numpy agar cepat diproses
    id1_array = df['id1'].values
    id2_array = df['id2'].values
    scores_array = df['score'].values
    
    # Cari nilai maksimum ID untuk menentukan ukuran lookup table
    max_id = int(np.max([np.max(id1_array), np.max(id2_array)]))
    num_embeddings = max_id + 1
    
    print(f"Max Card ID: {max_id}. Ukuran Tabel Embedding Murid: ({num_embeddings}, {embedding_dim})")
    
    # Inisialisasi Model Flax
    model = StudentEmbedding(num_embeddings=num_embeddings, features=embedding_dim)
    key = random.PRNGKey(42)
    dummy_input = jnp.array([0, 1])
    variables = model.init(key, dummy_input)
    
    # Inisialisasi Optimizer menggunakan Optax
    learning_rate = 0.01
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(variables)
    
    # Fungsi Training Step menggunakan JIT (Just-In-Time Compilation) agar sangat cepat di GPU/CPU
    @jit
    def train_step(variables, opt_state, batch_id1, batch_id2, batch_scores):
        def loss_fn(params):
            emb1 = model.apply(params, batch_id1)
            emb2 = model.apply(params, batch_id2)
            sim = cosine_similarity(emb1, emb2)
            # Mean Squared Error (MSE) Loss
            return jnp.mean((sim - batch_scores) ** 2)
        
        loss, grads = value_and_grad(loss_fn)(variables)
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        new_variables = optax.apply_updates(variables, updates)
        return new_variables, opt_state, loss

    # Konfigurasi Hyperparameter Training
    epochs = 5
    batch_size = 4096 # Batch besar karena model ini sangat kecil dan ringan
    dataset_size = len(id1_array)
    num_batches = dataset_size // batch_size
    
    print(f"Memulai Proses Distillation Training menggunakan JAX/Flax selama {epochs} Epochs...")
    
    for epoch in range(epochs):
        # Acak urutan dataset di awal setiap epoch
        perm = np.random.permutation(dataset_size)
        id1_shuffled = id1_array[perm]
        id2_shuffled = id2_array[perm]
        scores_shuffled = scores_array[perm]
        
        epoch_loss = 0.0
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i in range(0, dataset_size - batch_size, batch_size):
                # Siapkan Batch (jnp array akan otomatis pindah ke memori XLA/GPU)
                b_id1 = jnp.array(id1_shuffled[i : i+batch_size])
                b_id2 = jnp.array(id2_shuffled[i : i+batch_size])
                b_scores = jnp.array(scores_shuffled[i : i+batch_size])
                
                # Eksekusi step
                variables, opt_state, loss = train_step(variables, opt_state, b_id1, b_id2, b_scores)
                epoch_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                pbar.update(1)
                
        print(f"==> Rata-rata Loss Epoch {epoch+1}: {epoch_loss / num_batches:.4f}")
        
    # Ekstrak bobot hasil training untuk digunakan di model RL utama
    params = variables['params']
    embed_key = list(params.keys())[0] # Biasanya 'Embed_0'
    final_embeddings = params[embed_key]['embedding']
    
    # Konversi ke NumPy agar mudah disave/load nantinya
    np.save(output_weights, np.array(final_embeddings))
    print(f"\nTraining Distillation Selesai!")
    print(f"Bobot (weights) model RL Anda berhasil diekstrak dan disimpan di: {output_weights}")
    print(f"Bentuk array akhir: {final_embeddings.shape} (Siap diintegrasikan ke Model Utama)")

if __name__ == "__main__":
    train_student()
