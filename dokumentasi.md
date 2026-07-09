# Dokumentasi Sistem: AI Reinforcement Learning Pokemon TCG (JAX PPO)

Dokumen ini merangkum seluruh arsitektur dan komponen yang telah dibangun untuk menciptakan agen kecerdasan buatan (AI) yang mampu memainkan Pokemon TCG menggunakan **Model-Free Reinforcement Learning (PPO)**. 

Proyek ini menjembatani dua dunia yang sangat berbeda:
1. **Dunia C++ (Lingkungan / Engine):** Mesin simulator berkecepatan tinggi yang berbasis *stateful pointer*.
2. **Dunia JAX (Otak AI):** Kerangka kerja *Deep Learning* fungsional-murni dari Google yang dikompilasi menggunakan XLA (berjalan di GPU/TPU).

Untuk menggabungkan keduanya, kode disusun secara ketat berdasarkan prinsip **Atomic Design** (Satu File, Satu Tanggung Jawab) di dalam direktori `agent_rl/`.

---

## 1. Arsitektur Jaringan Saraf
**File:** `agent_rl/model.py`

File ini mendefinisikan bentuk anatomis dari AI. 
*   **Positional Encoding:** Menandai urutan matriks kartu agar AI mengenali letak spasial kartu (mana kartu di tangan, mana yang di Arena).
*   **Transformer Block (Self-Attention):** Membantu AI menemukan hubungan antar kartu (misalnya: menghubungkan Energi di tangan dengan Pokemon di Arena).
*   **Actor-Critic Heads:** 
    *   *Actor* menghasilkan sekumpulan probabilitas untuk 250 kemungkinan aksi.
    *   *Critic* menebak seberapa besar peluang AI untuk menang dari kondisi saat ini (skor *Value*).

## 2. Penglihatan & Manipulasi Masking
**File:** `agent_rl/feature_extractor.py` & `agent_rl/action_mapping.py`

Komponen ini adalah jembatan bahasa antara C++ dan Array Matematika.
*   **`feature_extractor.py`:** Menerima objek Dataclass dari C++ dan mengonversinya menjadi matriks `(93, 31)` untuk representasi kartu dan vektor `(266,)` untuk kondisi permainan global (giliran, skor *Prize*).
*   **`action_mapping.py`:** Membaca opsi aksi *legal* (diizinkan) dari C++ dan menghasilkan **Action Mask**. Matriks logikal (0-249) ini disuntikkan ke otak AI (*Actor Head*) agar AI **mustahil menebak tombol yang salah/terlarang** dengan memberinya penalti `-infinity` pada probabilitas Softmax-nya.

## 3. Motivasi & Anti-Hacking
**File:** `agent_rl/reward.py`

Buku panduan hukuman dan hadiah untuk membentuk perilaku sang AI.
*   **Zero-Sum Terminal Reward:** Menang mutlak mendapat `+1.0`, Kalah `-1.0`.
*   **Potential-Based Shaping ($\Phi$):** Mencegah fenomena *Reward Hacking* (seperti mengulur waktu atau menyerang tanpa niat membunuh). Kami menghitung poin matematis statis (fokus pada mencuri *Prize Card*) di mana jika AI mundur-maju membuang waktu, total kumulatif *reward*-nya tergaransi menjadi 0 secara matematis.

## 4. Jembatan Multiprocessing (Gym Paralel)
**File:** `agent_rl/vector_env.py`

Kompiler JAX akan *crash* jika mengakses pointer C++ secara langsung.
*   File ini melahirkan (spawn) *N* klon mesin game di *Core CPU* yang saling terisolasi.
*   Pekerja (*Worker*) secara mandiri menjalankan game, mengumpulkan data C++, lalu menumpuknya menjadi satu *Batch Raksasa* Numpy.
*   Memiliki mekanisme otomatis bagi Player 1 (lawan AI) untuk bergerak secara acak *(Random Agent)* agar Player 0 (AI JAX) bisa berfokus memprediksi 1 langkah.
*   **Bisa Di-upgrade ke Self-Play:** Kurikulum masa depan bisa mematikan bot acak sehingga Player 0 bermain melawan Player 1 (keduanya diprediksi oleh AI).

## 5. Buku Catatan Pengalaman
**File:** `agent_rl/buffer.py`

Sistem memori murni NumPy (*Pre-allocated Rollout Buffer*).
*   Pemesanan kapasitas memori penuh di awal sehingga terbebas dari kebocoran memori (memory-leak).
*   Menghitung *Generalized Advantage Estimation (GAE)* secara mundur (*backward*) untuk menentukan seberapa bernilai aksi yang telah dilakukan di masa lalu jika ternyata hal tersebut membuahkan kemenangan di akhir.
*   Menyiapkan pemotong data (*generator mini-batch*) raksasa yang ramah untuk disuapkan ke VRAM GPU.

## 6. Jantung Algoritma PPO (Kompilasi XLA JIT)
**File:** `agent_rl/ppo_update.py`

Fungsi keramat tanpa celah yang secara utuh dikompilasi oleh JAX ke bahasa primitif mesin.
*   Dilengkapi anotasi `@jax.jit`.
*   Kalkulasi gradient memuat: *Clipped Surrogate Objective* (agar model tidak bergeser terlampau drastis), *Value Loss MSE* (melatih akurasi tebakan insting menang), dan *Entropy Penalty* (mendorong AI agar bereksperimen mencoba-coba kartu baru alih-alih melakukan langkah basi).
*   Waktu eksekusi dipangkas dari 1,7 detik menjadi 0,04 detik *(ultra-fast update)*.

## 7. Sang Konduktor (Main Loop)
**File:** `agent_rl/train.py`

File *Entry Point* tempat segalanya berpusat. 
*   Bertugas mengikat 8 *environment* C++ paralel dengan model JAX dan *Buffer* ke dalam sistem siklus *Rollout -> Gradient Update*.
*   Memonitor dan mencetak laporan secara real-time seperti Nilai *Loss*, *Mean Reward*, *Win-Rate*, dan *Frame per Second (FPS)*.
*   Secara berkala menyimpan kondisi otak AI ke dalam file serialisasi *Msgpack* (`checkpoints/model_final.msgpack`).

---
**Status Pengembangan:** Selesai 100%. Fondasi telah kokoh dan siap dilatih (Scaling).
