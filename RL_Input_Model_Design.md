# Desain Input Model AI untuk Pokémon TCG (State-of-the-Art)

Dokumen ini menjelaskan spesifikasi ekstraksi fitur (*Feature Engineering*) untuk melatih model Reinforcement Learning (seperti PPO) pada *engine* game Pokémon TCG. Desain ini telah dikalibrasi sedemikian rupa agar sangat optimal, terhindar dari bias, dan memiliki pemahaman mendalam tentang mekanik *game* seperti kelemahan elemen dan ketersediaan aksi.

## Konsep Utama
Alih-alih menggunakan *Multi-hot array* yang mencampuradukkan data, sistem ini merombak data menjadi kumpulan **Padded Sequence Array (untuk Embedding)** dan **Matriks Skalar Normalisasi**.

---

## 1. Ekstraksi Fitur Kartu (Card Features)

Tugas utama bagian ini adalah mencatat "Kartu apa saja yang sedang terlibat di arena dan di tangan". Semua ini direpresentasikan sebagai deretan angka *Integer (Card ID)* berukuran maksimal `60` (ukuran sebuah Deck TCG).

* **`my_hand`** `(60,)` : Deretan ID kartu di tangan kita. (Misal: `[10, 45, 12, 0, 0, ...]`)
* **`my_discard`** `(60,)` : Deretan ID kartu di tumpukan sampah kita.
* **`opp_discard`** `(60,)` : Deretan ID kartu di tumpukan sampah lawan.
* **`my_active_id`** `(1)` : ID kartu Pokemon Active kita.
* **`opp_active_id`** `(1)` : ID kartu Pokemon Active lawan.

> **Cara Penggunaan di Neural Network:** Array di atas akan dimasukkan ke dalam **Embedding Layer** (misal dengan dimensi `32`) untuk mengekstrak makna dari masing-masing kartu, lalu dilakukan *Pooling* (Sum/Mean).

---

## 2. Ekstraksi Fitur Global (Global Features)

Menyimpan 10 informasi yang tidak terikat pada kartu Pokemon tertentu di arena. Bentuknya adalah matriks `(10,)`.

| Indeks | Fitur | Penjelasan / Normalisasi |
| :--- | :--- | :--- |
| `0` | `turn_normalized` | Giliran saat ini `(dibagi 100.0)` |
| `1` | `action_count` | Jumlah aksi dalam satu turn `(dibagi 50.0)` |
| `2` | `is_first_player` | `1.0` jika kita pemain pertama |
| `3` | `supporter_played` | `1.0` jika jatah Supporter sudah dipakai |
| `4` | `energy_attached` | `1.0` jika jatah pasang energi manual sudah dipakai |
| `5` | `stadium_id` | ID Kartu Stadium (Untuk di-Embedding) |
| `6` | `my_deck_fraction` | Sisa kartu di deck kita `(deckCount / 60.0)` |
| `7` | `opp_deck_fraction` | Sisa kartu di deck lawan `(deckCount / 60.0)` |
| `8` | `my_prize_fraction`| Sisa prize kita `(prizeCount / 6.0)` |
| `9` | `opp_prize_fraction`| Sisa prize lawan `(prizeCount / 6.0)` |

---

## 3. Ekstraksi Fitur Arena (Board Features) - 31 Dimensi

Ini adalah jantung dari State AI. Matriks berukuran **`(2, 6, 31)`**.
* **Dimensi 1 (`2`)**: Indeks Pemain (Kita dan Lawan).
* **Dimensi 2 (`6`)**: Slot Pokémon (0 = Active, 1-5 = Bench).
* **Dimensi 3 (`31`)**: Atribut taktis super detail per Pokémon.

Berikut rincian dari **31 Dimensi** tersebut:

### A. Fitur Kategori (Untuk Embedding)
| Indeks | Fitur | Tipe | Penjelasan |
| :--- | :--- | :--- | :--- |
| `0` | `card_id` | Integer | ID Pokemón di slot ini. |
| `1` | `tool_id` | Integer | ID Item/Tool yang terpasang. |
| `2` | `pre_evolution_id` | Integer | ID Wujud sebelum berevolusi (Penting untuk deteksi target *Devolve*). |

### B. Fitur Dasar & Kerusakan
| Indeks | Fitur | Tipe | Penjelasan |
| :--- | :--- | :--- | :--- |
| `3` | `is_present` | Float `(0/1)` | `1.0` jika slot terisi. |
| `4` | `hp_fraction` | Float | Persentase sisa darah `(hp / maxHp)`. |
| `5` | `damage_counters` | Float | Angka kerusakan absolut `(maxHp - hp) / 10.0`. Sangat penting jika efek serangan berhitung dengan kerusakan angka tetap. |
| `6` | `appear_this_turn`| Float `(0/1)` | `1.0` jika baru turun giliran ini. |

### C. Status Kondisi (Hanya Slot Active)
Indeks `7` hingga `11` berturut-turut mengecek apakah Pokemon terkena: `poisoned`, `burned`, `asleep`, `paralyzed`, dan `confused`. Bernilai `1.0` jika iya.

### D. Normalisasi Energi (Requirement Satisfaction Ratio)
**Indeks `12` hingga `23` (12 Elemen):** Mewakili jumlah energi yang menempel.
Alih-alih menyajikan angka mentah (misal: 3 energi), nilai ini **dinormalisasi berdasarkan energi maksimal yang dibutuhkan oleh serangan kartu tersebut**.
* *Contoh:* Charizard butuh 4 energi Api. Di arena menempel 2 energi Api.
* *Nilai Input (Indeks `12+tipe_api`):* `2 / 4 = 0.5`.
* Ini membuat AI memiliki nalar universal bahwa angka `1.0` berarti "Kebutuhan energinya sudah terpenuhi!".

### E. Kolom Aksi / Tombol Menyala (Action Readiness)
Alih-alih membiarkan AI menebak energi, kita memberikan sinyal hijau secara eksplisit jika aksinya sah digunakan di giliran ini:
| Indeks | Fitur | Penjelasan |
| :--- | :--- | :--- |
| `24` | `attack_1_ready` | `1.0` jika Serangan 1 sah dilakukan saat ini. |
| `25` | `attack_2_ready` | `1.0` jika Serangan 2 sah dilakukan saat ini. |
| `26` | `ability_1_ready` | `1.0` jika Kemampuan 1 sah digunakan. |
| `27` | `ability_2_ready` | `1.0` jika Kemampuan 2 sah digunakan. |
| `28` | `can_retreat` | `1.0` jika Pokemon memiliki energi yang cukup untuk mundur. |

### F. Type Matchup / Benturan Tipe
Indeks khusus untuk mendidik insting kalkulator *Damage* pada AI. Hanya dihitung untuk interaksi **Active vs Active**.
| Indeks | Fitur | Penjelasan |
| :--- | :--- | :--- |
| `29` | `is_hitting_weakness` | `1.0` jika elemen serangan kita mengenai kelemahan lawan (Sinyal bahaya ekstra jika dari sudut pandang lawan). |
| `30` | `is_hitting_resistance` | `1.0` jika serangan kita akan teredam resistensi lawan. |

---

## 4. Total Kalkulasi Matriks Neural Network
Jika data mentah di atas dirajut di dalam layer *JAX/Flax*:
1. Matriks Hand & Discard dimasukkan ke `nn.Embed` $\rightarrow$ diubah menjadi representasi makna dan digabung (Fusion).
2. Matriks Board memecah indeks `[0-2]` menjadi `nn.Embed`, lalu menggabungkannya kembali dengan `28` sisa angkanya.
3. Seluruh bagian *Board*, *Global*, dan *Cards* digabungkan ke dalam sebuah *Main MLP (Multi-Layer Perceptron)*.
4. Output bermuara pada **Policy Head** (Distribusi probabilitas `200` daftar aksi, disaring melalui *Action Masking* 100% akurat), dan **Value Head** (Prediksi nilai posisi -1.0 hingga 1.0).
