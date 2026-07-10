# Daftar Tugas (Task List) - Pendekatan Atomic Design

Dokumen ini memetakan tugas pengembangan berdasarkan **file-file individual** yang harus dibuat atau direfaktor. Dengan pendekatan *Atomic*, setiap file memiliki tanggung jawab tunggal *(Single Responsibility Principle)* sehingga kode lebih rapi, terisolasi, dan mudah di- *debug*.

---

### 1. `agent_rl/model.py` (Domain: Arsitektur Neural Network JAX)
File ini murni berisi arsitektur struktural AI. Tidak ada logika *game* di sini.
- [x] Buat class `PositionalEncoding` (Menyuntikkan ID posisi ke urutan kartu).
- [x] Buat class `TransformerBlock` (Self-Attention + FFN + LayerNorm).
- [x] Buat class `PokemonAgent` (Model Utama):
  - Eksekusi *Transformer*, *Slicing*, *Flattening*, dan *Fusion*.
  - Eksekusi **Actor Head** (Logits array dimensi 250 + Action Masking `logits - 1e9`).
  - Eksekusi **Critic Head** (Value/Peluang Menang dengan batas `[-1.0, 1.0]`).

### 2. `agent_rl/action_mapping.py` (Domain: Standarisasi Output)
File **baru** untuk mengisolasi logika konversi *action* (memisahkan kekacauan *if/else* dari *feature extractor*).
- [x] Buat konstanta/konfigurasi pemetaan baku 250 aksi (misal: `ACTION_END_TURN = 180`).
- [x] Buat fungsi `create_action_mask(select_data)`: Menerima list `Option` C++ dan mengembalikan *Numpy Boolean Array* berukuran 250 yang menandakan legalitas aksi saat ini.
- [x] Buat fungsi `decode_action(action_index)`: Mengonversi kembali indeks JAX (0-249) yang dipilih AI menjadi format list `[int]` yang bisa dibaca C++ (`search_step`).

### 3. `agent_rl/feature_extractor.py` (Domain: Standarisasi Input)
File ini khusus bertugas mengubah JSON/Objek API menjadi *Numpy Array*.
- [x] Buat logika pembentukan matriks `Card_Embedding_Sequence` berdimensi `(93, 31)` (Catatan: diubah dari 60 menjadi 31 karena 3 ID akan di-embed di dalam model JAX. Terdiri dari 3 ID + 28 Skalar) untuk menyusun kartu di zona *Hand, Discard, Board*, dan *Stadium*.
- [x] Panggil fungsi `create_action_mask` dari `action_mapping.py`.
- [x] Susun `Global_State` dengan menggabungkan 16 fitur dasar + 250 *Action Mask* menjadi vektor dimensi `266`.

### 4. `agent_rl/reward.py` (Domain: Kalkulasi Fungsi Potensial)
File **baru** untuk mengeksekusi logika *anti-hacking* yang dirancang di `reward_sistem.md`.
- [x] Buat fungsi `calc_potential(state, your_index)` yang menghitung skor *Prize, HP,* dan *Deck Out*.
- [x] Buat fungsi `calculate_step_reward(old_state, new_state, your_index, is_terminal)` yang menghasilkan perhitungan mutlak (*Shaping + Time Penalty + Terminal*).

### 5. `agent_rl/vector_env.py` (Domain: Jembatan Eksekusi C++)
File **baru** untuk membungkus `api.py` agar bisa berjalan di *multiprocessing* tanpa merusak JAX.
- [x] Buat arsitektur *Sub-Process Worker* yang menginisiasi `lib.AgentStart()` secara mandiri di setiap *core* CPU.
- [x] Buat fungsi `step_async(actions)` untuk mendistribusikan array aksi ke masing-masing *worker*.
- [x] Buat fungsi `step_wait()` untuk mengumpulkan dan mem-*batch* (stacking) obversasi, reward, dan status `done` dari semua *worker* menjadi satu *Numpy Array* utuh berdimensi `(N, ...)`.

### 6. `agent_rl/buffer.py` (Domain: Penyimpanan Pengalaman)
File **baru** untuk mencatat jejak permainan sebelum diserahkan ke GPU.
- [x] Buat class `RolloutBuffer` murni berbasis NumPy.
- [x] Siapkan memori berkapasitas pre-alokasi `(Batch_Size, Dimensi)` untuk menyimpan: `states`, `actions`, `log_probs`, `rewards`, dan `values`.
- [x] Buat generator batch otomatis untuk memecah memori besar menjadi *mini-batches* saat *training*.

### 7. `agent_rl/ppo_update.py` (Domain: Kalkulasi Gradien)
File **baru** yang murni berisi rumus matematika pembaruan bobot *(weight updates)* JAX.
- [x] Tulis fungsi komputasi *Generalized Advantage Estimation* (GAE). (Diintegrasikan di `buffer.py`)
- [x] Tulis *Loss Function* untuk PPO (*Clipped Surrogate, Value Loss MSE, Entropy Penalty*).
- [x] Bungkus keseluruhan fungsi ini dengan anotasi `@jax.jit` agar kompilasi murni dikerjakan oleh XLA GPU/TPU.

### 8. `agent_rl/train.py` (Domain: Orkestrasi / Loop Utama)
File utama (*Entry Point*) yang akan dieksekusi saat proses *training*.
- [ ] Inisialisasi Model, `vector_env`, `buffer`, dan *Optimizer* (Optax).
- [ ] Buat *Main Training Loop*:
  - Fase Pengumpulan (Menggerakkan `vector_env` dan mencatat ke `buffer`).
  - Fase Update (Melempar data `buffer` ke fungsi di `ppo_update.py`).
- [ ] Integrasikan fitur penyimpanan bobot (checkpoint) dan pemantauan metrik latihan (misal: *mean reward*, *win-rate*).
