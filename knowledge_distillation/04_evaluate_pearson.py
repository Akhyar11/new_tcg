import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def evaluate_correlation():
    csv_path = "card_similarity_scores.csv"
    npy_path = "student_embeddings_32d.npy"
    
    print("1. Membaca target Guru (384D) dari CSV...")
    df = pd.read_csv(csv_path)
    
    id1_array = df['id1'].values
    id2_array = df['id2'].values
    teacher_scores = df['score'].values
    
    print("2. Membaca bobot Murid (32D)...")
    student_embeddings = np.load(npy_path)
    
    # Normalisasi bobot Murid
    norms = np.linalg.norm(student_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9 # Mencegah pembagian dengan nol
    student_norm = student_embeddings / norms
    
    print("3. Menghitung similarity dari Murid secara bulk (Matrix Dot Product)...")
    student_sim_matrix = np.dot(student_norm, student_norm.T)
    
    # Mengambil nilai skor murid berdasarkan index pasangan (id1, id2)
    # NumPy advanced indexing ini sangat cepat untuk jutaan baris
    student_scores = student_sim_matrix[id1_array, id2_array]
    
    print("4. Menghitung Pearson dan Spearman Correlation (Evaluasi Statistik)...")
    pearson_corr, pearson_pval = pearsonr(teacher_scores, student_scores)
    spearman_corr, spearman_pval = spearmanr(teacher_scores, student_scores)
    
    print("\n" + "="*50)
    print("=== HASIL EVALUASI KNOWLEDGE DISTILLATION ===")
    print("="*50)
    print(f"Total Sampel         : {len(teacher_scores):,} pasang kartu")
    print(f"Pearson Correlation  : {pearson_corr:.4f} (Rentang: -1 s.d 1)")
    print(f"Spearman Correlation : {spearman_corr:.4f} (Rentang: -1 s.d 1)")
    print("="*50)
    
    print("\n[Interpretasi Hasil]:")
    if pearson_corr > 0.90:
        print("⭐⭐⭐⭐⭐ (SANGAT TINGGI) - Murid 32D berhasil mereplika hampir sempurna otak semantik Guru 384D.")
    elif pearson_corr > 0.80:
        print("⭐⭐⭐⭐ (TINGGI) - Murid 32D menangkap sebagian besar makna dengan sangat baik.")
    elif pearson_corr > 0.70:
        print("⭐⭐⭐ (CUKUP) - Distilasi berhasil dan bisa digunakan, walau ada detail minor yang hilang.")
    else:
        print("⭐ (RENDAH) - Ada banyak kompresi informasi. Mungkin perlu dimensi lebih tinggi (misal: 64D) atau Epoch lebih lama.")

if __name__ == "__main__":
    evaluate_correlation()
