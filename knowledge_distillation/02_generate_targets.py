import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def generate_similarity_csv():
    csv_path = "../agent_rl/EN_Card_Data.csv"
    output_csv = "card_similarity_scores.csv"
    model_dir = "./teacher_model"

    print("Membaca database kartu...")
    df = pd.read_csv(csv_path)

    # Mengelompokkan data berdasarkan Card ID, karena 1 kartu (1 Card ID) bisa punya lebih dari 1 baris (karena beda attack/move)
    print("Menggabungkan baris untuk kartu dengan ID yang sama...")
    grouped = df.groupby("Card ID")
    
    card_ids = []
    card_texts = []

    columns_to_read = [
        "Card Name", "Expansion", "Collection No.", "Stage (Pokémon)/Type (Energy and Trainer)",
        "Rule", "Category", "Previous stage", "HP", "Type", "Weakness", "Resistance (Type)",
        "Retreat", "Move Name", "Cost", "Damage", "Effect Explanation"
    ]

    for card_id, group in grouped:
        # Ambil info dasar dari baris pertama (karena stat dasar sama untuk ID yang sama)
        base_row = group.iloc[0]
        
        text_parts = []
        # Tambahkan info dasar
        for col in columns_to_read[:12]: # Kolom Info Dasar
            val = base_row[col]
            if pd.notna(val) and val != "n/a":
                text_parts.append(f"{col}: {val}")
        
        # Tambahkan info Move/Attack dari SEMUA baris untuk ID ini
        moves_texts = []
        for _, row in group.iterrows():
            move_parts = []
            for col in columns_to_read[12:]: # Kolom Moves
                val = row[col]
                if pd.notna(val) and val != "n/a":
                    move_parts.append(f"{col}: {val}")
            if move_parts:
                moves_texts.append(" | ".join(move_parts))
        
        if moves_texts:
            text_parts.append("Moves -> " + " || ".join(moves_texts))
            
        full_text = ", ".join(text_parts)
        card_ids.append(card_id)
        card_texts.append(full_text)

    print(f"Total kartu unik: {len(card_ids)}")

    # Load Model Xenova (MiniLM)
    print("Memuat model teacher...")
    # Jika folder lokal tidak ada, fallback download dari HF
    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Ekstrak Embedding dalam batch
    batch_size = 64
    all_embeddings = []

    print("Mengekstrak teks menjadi embedding vektor...")
    with torch.no_grad():
        for i in tqdm(range(0, len(card_texts), batch_size)):
            batch_texts = card_texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # Mean Pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize embedding agar bisa dihitung dot product (cosine similarity)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    print("Menghitung skor kemiripan (Cosine Similarity)...")
    # Karena sudah dinormalisasi, Cosine Similarity = Dot Product
    similarity_matrix = torch.mm(all_embeddings, all_embeddings.t())

    print("Menyusun data CSV untuk id1, id2, score...")
    # Untuk menghemat tempat dan menghilangkan duplikasi, kita bisa hanya mengambil kombinasi unik (i <= j)
    # Tapi untuk fleksibilitas JAX Dataloader nanti, kita simpan seluruh matrix atau separuh atasnya.
    # Kita simpan seluruhnya saja (id1, id2, score).
    
    # Flatten matrix
    # Menggunakan generator atau list comprehension untuk membuat DataFrame
    n = len(card_ids)
    
    # Kita gunakan pendekatan efisien untuk membuat dataframe besar (~4 juta baris)
    id1_list = []
    id2_list = []
    score_list = []
    
    # Optimasi pembuatan list (~4 juta pasang)
    for i in tqdm(range(n)):
        for j in range(n):
            id1_list.append(card_ids[i])
            id2_list.append(card_ids[j])
            score_list.append(round(similarity_matrix[i, j].item(), 4))

    out_df = pd.DataFrame({
        "id1": id1_list,
        "id2": id2_list,
        "score": score_list
    })

    print(f"Menyimpan ke {output_csv}...")
    out_df.to_csv(output_csv, index=False)
    print(f"Selesai! File disimpan dengan jumlah baris: {len(out_df)}")

if __name__ == "__main__":
    generate_similarity_csv()
