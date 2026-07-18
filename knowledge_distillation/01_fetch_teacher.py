import os
import torch
from transformers import AutoTokenizer, AutoModel

def fetch_and_save_teacher():
    # Xenova/all-MiniLM-L6-v2 adalah model embedding yang populer dan sangat efisien.
    # Kita menggunakan model aslinya dari sentence-transformers (karena Xenova memporting dari sini)
    # yang 100% identik representasi matematisnya.
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    save_dir = "./teacher_model"
    
    print(f"Mendownload Tokenizer dari {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Mendownload Model dari {model_id}...")
    # Kita menggunakan PyTorch di sini karena ekosistem HuggingFace lebih kuat di PyTorch untuk pengunduhan.
    # Ingat: Model ini HANYA dipakai offline untuk menyusun skor target (Distillation Label).
    # Model RL (Murid) nantinya akan murni 100% menggunakan JAX/Flax sesuai permintaan.
    model = AutoModel.from_pretrained(model_id)
    
    print(f"Menyimpan model ke {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
    # Uji Coba Model
    print("\n--- Uji Coba Model Teacher ---")
    test_text = "Draw 2 cards."
    print(f"Teks input: '{test_text}'")
    
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean Pooling (karena MiniLM menggunakan mean pooling untuk sentence embedding)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    print(f"Bentuk Embedding Output: {embeddings.shape}")
    print(f"Proses pengambilan model selesai dan berhasil disimpan secara lokal di '{save_dir}'.")

if __name__ == "__main__":
    fetch_and_save_teacher()
