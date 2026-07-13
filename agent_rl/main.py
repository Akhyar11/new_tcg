import random
import numpy as np
import jax
import jax.numpy as jnp
from feature_extractor import extract_features
from model import PokemonAgent

# Inisialisasi model (contoh dengan random weights sementara)
model = PokemonAgent(num_actions=200)
key = jax.random.PRNGKey(0)

# Dummy state untuk menginisialisasi parameter model
dummy_features = {
    "global": jnp.zeros(10),
    "board": jnp.zeros((2, 6, 20)),
    "hand": jnp.zeros(250)
}
dummy_mask = jnp.zeros(200, dtype=bool)
params = model.init(key, dummy_features, dummy_mask)

def agent(obs_dict: dict) -> list[int]:
    """
    Fungsi agen utama yang akan dipanggil oleh Kaggle Engine.
    """
    # 1. Jika engine meminta deck di awal permainan
    if obs_dict.get('select') is None:
        # TODO: Return susunan deck awal 60 kartu (gunakan bacaan dari CSV)
        import os
        file_path = "deck.csv"
        if not os.path.exists(file_path):
            file_path = "/kaggle_simulations/agent/" + file_path
        with open(file_path, "r") as file:
            deck = [int(x) for x in file.read().split("\n")[:60]]
        return deck
        
    # 2. Ekstrak JSON obs menjadi JAX Array dan Action Mask
    features, action_mask = extract_features(obs_dict)
    
    # 3. Prediksi aksi menggunakan JAX Model
    # Untuk inferensi, kita ubah ke jnp array
    j_features = {k: jnp.array(v) for k, v in features.items()}
    j_mask = jnp.array(action_mask)
    
    logits = model.apply(params, j_features, j_mask)
    
    # 4. Pilih aksi dengan nilai terbesar (argmax) atau secara probabilitas
    # Karena ini simulasi lokal sementara belum ditraining, kita sample aja
    max_count = obs_dict['select']['maxCount']
    valid_options = np.where(action_mask)[0]
    
    # (Placeholder) Jika opsi valid < max_count, return semua
    if len(valid_options) > 0:
        # Untuk saat ini kita return acak dari yang sah untuk menghindari error 
        # sampai model kita hasilkan logit yang masuk akal
        selected = random.sample(list(range(len(obs_dict['select']['option']))), max_count)
    else:
        selected = []
        
    return selected
