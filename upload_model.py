import kagglehub
import os

# Ganti dengan path ke direktori yang berisi file model Anda (file .msgpack)
LOCAL_MODEL_DIR = 'checkpoints'

# Nama Model Utama (Slug) yang akan muncul di Kaggle
MODEL_SLUG = 'pokemon-tcg-ppo-jax' 

# Nama variasi model (Bisa digunakan untuk membedakan eksperimen, misal: 'v1-random-bot')
VARIATION_SLUG = 'default' 

print(f"Memulai proses unggah direktori '{LOCAL_MODEL_DIR}' ke Kaggle Models...")
print(f"Target Handle: akhyarsafrudin/{MODEL_SLUG}/jax/{VARIATION_SLUG}")

try:
    # Memastikan direktori checkpoints ada
    if not os.path.exists(LOCAL_MODEL_DIR):
        print(f"Error: Direktori '{LOCAL_MODEL_DIR}' tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan agent_rl/train.py minimal sekali hingga checkpoint terbentuk.")
        exit(1)

    kagglehub.model_upload(
      handle = f"akhyarsafrudin/{MODEL_SLUG}/jax/{VARIATION_SLUG}",
      local_model_dir = LOCAL_MODEL_DIR,
      version_notes = 'Initial JAX PPO Model Checkpoints'
    )
    print("\n✅ Upload model ke Kaggle berhasil!")
except Exception as e:
    print(f"\n❌ Gagal mengunggah model: {e}")
    print("Pastikan Anda sudah mengonfigurasi kredensial API Kaggle (Kaggle.json) di komputer Anda.")
