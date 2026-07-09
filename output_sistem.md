# Arsitektur Output Model (Action Space) untuk Pokemon TCG RL

Mendesain ruang aksi (*action space*) untuk *Trading Card Game* (TCG) seperti Pokemon adalah salah satu tantangan terbesar dalam arsitektur Reinforcement Learning. Hal ini dikarenakan aksi dalam game kartu bersifat sangat **dinamis** dan **berubah-ubah** bergantung pada *state* dan konteks `SelectData` dari *engine*.

Ada dua pendekatan utama yang bisa diimplementasikan untuk *output* model kita. Kita perlu menetapkan rancangan yang paling tepat sebelum menulis kode untuk layer PPO / Output JAX.

---

## Pendekatan 1: Fixed Action Space (Vektor Statis + Action Masking)
Ini adalah pendekatan klasik (Discrete PPO standar) yang paling mudah diimplementasikan di JAX. Model **selalu** menghasilkan vektor *logit* dengan ukuran tetap (misal: `Num_Actions = 250`).

Kita mendefinisikan sebuah **Canonical Action Mapping** (Pemetaan Aksi Baku). Saat `api.py` (C++) memberikan daftar `option` yang legal, kita mengubah *opsi* tersebut ke dalam indeks baku (0 - 249) untuk mengisi nilai `1` pada *Action Mask*, dan menaruh nilai `0` untuk sisanya.

**Contoh Pemetaan (Draft ~250 Aksi):**
*   `0 - 59`: **Aksi Hand.** (Pilih/Mainkan/Buang kartu di *Hand* berdasarkan letak indeks array 0 s/d 59).
*   `60 - 119`: **Aksi Deck.** (Pilih kartu dari Deck saat terkena efek *search*, indeks 0 s/d 59).
*   `120 - 179`: **Aksi Discard/Lost Zone.** (Pilih kartu dari tempat sampah, indeks 0 s/d 59).
*   `180`: **End Turn.**
*   `181`: **Retreat.**
*   `182 - 185`: **Serang (Attack 1, 2, 3, 4)** dari Pokemon Aktif kita.
*   `186 - 197`: **Gunakan Ability / Kemampuan** (2 slot ability $\times$ 6 slot arena [1 Active + 5 Bench]).
*   `198 - 209`: **Pilih Target Area Kita** (Menargetkan Pokemon Active kita, atau Bench 1-5).
*   `210 - 221`: **Pilih Target Area Lawan** (Menargetkan Pokemon Active musuh, atau Bench 1-5).
*   `222`: **Pilih Yes (Konfirmasi).**
*   `223`: **Pilih No (Tolak).**
*   `224 - 230`: **Pilih Tipe Energi** (Grass, Fire, Water, dst).

*   **Kelebihan:** Sangat mudah dipasang ke algoritma PPO standar JAX. Proses komputasi pada *layer* akhir sangat ringan (hanya satu *Dense Layer* 256 $\rightarrow$ 250).
*   **Kelemahan:** Membutuhkan kode Python (*parser*) yang sangat kaku (`if/else`) di fungsi *Feature Extraction* untuk memetakan bolak-balik antara C++ `Option` objek dan Index Model.

---

## Pendekatan 2: Parametrized Action Space (Pointer / Option Embedding Mechanism)
Daripada memaksa model mengeluarkan vektor statis berukuran 250, model merespons murni berdasarkan *pilihan* yang diberikan secara dinamis oleh lingkungan. Ini adalah teknik *State-Of-The-Art* yang digunakan oleh bot kompetitif tinggi (mirip AlphaStar).

**Mekanismenya:**
1. *Engine* C++ `SelectData` memberikan array `option` (Misal saat ini ada 5 opsi yang legal).
2. Kita menerjemahkan masing-masing dari ke-5 opsi tersebut menjadi sebuah **Vektor Embedding** berukuran `128` (berdasarkan `Card_Id`, tipe aksi, dsb).
3. Model memproses `Global_State` dan `Board_Sequence` lalu menghasilkan sebuah **Context_Vector** berukuran `128`.
4. Lakukan operasi *Dot Product* antara **Context_Vector** dan kelima **Option Embedding** untuk menghasilkan 5 skor (*logits*).
5. Terapkan algoritma *Softmax* langsung pada 5 skor tersebut untuk memilih aksi.

*   **Kelebihan:** 100% Fleksibel. Model tetap luwes merespons baik ketika opsinya hanya 2 (Yes/No) maupun ketika ada 50 (Memilih kartu). Kita tidak perlu *Action Masking* berukuran besar. Tidak mungkin terjadi tabrakan indeks.
*   **Kelemahan:** Arsitektur *tensor logic* di JAX menjadi lebih kompleks. Kita perlu merancang jaringan `Option_Embedding_Layer` khusus agar model "mengerti" arti dari setiap opsi.

---

## Tantangan Khusus: Pemilihan Multi-Aksi (`maxCount > 1`)
Di dalam Pokemon TCG, ada aksi yang mengharuskan kita memilih lebih dari satu item sekaligus. Misalnya instruksi dari kartu Profesor: *"Buang hingga 2 kartu dari tanganmu"*.

C++ *engine* akan mengeset parameter `minCount=0`, `maxCount=2`, dan menyajikan `option` daftar kartu yang ada di tangan. Jika model kita menggunakan JAX (yang dirancang memprediksi `argmax` 1 aksi terbaik), bagaimana cara kita mengatasi opsi ganda ini?

Dua pendekatan yang bisa diambil:
1.  **Auto-Regressive (Bermain berulang dalam pikiran):**
    Model tidak langsung memilih 2. Model memilih aksi pertama (1 buah). Kemudian lingkungan C++ tidak dilanjutkan maju (*step*), melainkan fungsi internal Python secara artifisial mengembalikan *state* yang sama namun tanpa opsi yang telah dipilih. Model lalu *forward pass* lagi untuk memilih aksi kedua. Setelah lengkap (atau model memilih "Berhenti/End Selection"), baru kita kirim `[pilihan1, pilihan2]` ke dalam fungsi `search_step` C++.
2.  **Top-K Selection (Paling mudah):**
    Kita langsung melihat probabilitas dari distribusi *Softmax* output model, lalu mengambil $K$ opsi dengan *confidence* tertinggi untuk dieksekusi secara instan. Kelemahannya, pilihan ke-2 mungkin secara strategis tidak nyambung dengan pilihan ke-1. Pendekatan Auto-Regressive jauh lebih baik.

---

### Saran Pengambilan Keputusan
Untuk membangun **baseline pertama** (*Minimum Viable Agent*), disarankan memulai dengan **Pendekatan 1 (Fixed Action Space)** dan **Pendekatan Top-K** agar sistem *end-to-end* PPO dapat di-train sesegera mungkin. 

Setelah baseline terbukti dapat bermain tanpa *crash* dan berhasil mengaplikasikan fungsi *Reward Shaping* yang kita rancang, barulah arsitektur JAX di- *upgrade* menggunakan **Pendekatan 2 (Parametrized)**.
