# Sistem Reward yang Solid untuk Pokemon TCG (Anti-Reward Hacking)

Dalam merancang sistem reward untuk agen Reinforcement Learning (RL) yang memainkan Pokemon TCG, hambatan terbesar adalah potensi **Reward Hacking**. Agen AI mungkin menemukan celah eksploitasi (loop) untuk memaksimalkan skor per langkah (dense reward) tanpa mempedulikan kemenangan akhir (win condition).

Contoh eksploitasi yang sering terjadi:
1. **Retreat / Switch Farming:** Terus menerus me-retreat Pokemon tanpa henti untuk mendapatkan reward "melakukan aksi" atau "selamat dari serangan".
2. **Damage / Heal Looping:** Sengaja menyerang musuh berkali-kali tapi tidak mematikan, lalu membiarkan musuh melakukan healing agar agen bisa mencetak skor "damage dealt" lagi di giliran berikutnya secara infinit.
3. **Stalling (Mengulur Waktu):** Jika agen merasa akan kalah dan penalti kekalahan itu statis, agen mungkin sengaja mengulur-ulur waktu (pass turn) sebisa mungkin agar tidak kalah, menyebabkan game sangat panjang dan konvergensinya terganggu.

Untuk memastikan **100% Anti-Reward Hacking**, kita wajib mengombinasikan **Zero-Sum Sparse Rewards** dengan teknik **Potential-based Reward Shaping**.

## 1. Zero-Sum Terminal Reward (Penentu Utama)
Fokus mutlak AI haruslah memenangkan permainan. Berikan reward terbesar di akhir permainan (Terminal State). Karena Pokemon TCG adalah game PvP, kita gunakan *Zero-Sum* format.

Di akhir permainan (saat `State.result != -1` pada `api.py`):
*   **Win (Menang):** `+1.0` (Jika `State.result == yourIndex`)
*   **Loss (Kalah):** `-1.0` (Jika `State.result == 1 - yourIndex`)
*   **Draw (Seri):** `0.0` (Jika `State.result == 2`)

> Agen tidak boleh mendapat nilai akumulasi yang bisa melebihi batas `[-1.0, 1.0]` dari reward langkah per langkah, agar reward utamanya tidak tereduksi / diabaikan oleh agen.

## 2. Potential-Based Reward Shaping (Anti-Farming)
Alih-alih memberikan poin mutlak untuk aksi (misal: "+0.1 jika nge-damage musuh"), kita mendefinisikan sebuah **Potential Function ($\Phi$)** untuk menilai seberapa bagus state permainan saat ini secara statis.
Lalu reward dihitung sebagai turunan / selisih:
`R = Potential(State_sekarang) - Potential(State_sebelumnya)`

**Keuntungan Matematis:** 
Dengan menggunakan *selisih* potensi (potential difference), segala bentuk loop (seperti menyerang lalu musuh heal, mundur maju retreat) hanya akan menaikkan dan menurunkan potensial secara bolak-balik. Akumulasi (sum) reward selama loop tersebut akan bernilai mutlak `0`. Ini menghilangkan bug reward farming secara total!

### Rumus Potensial ($\Phi$) untuk TCG:
Kita mengekstrak `State` dari `api.py`:
1.  **Prize Card Potential:** Kemenangan utama di Pokemon TCG adalah Prize Card.
    `My_Prize_Taken = 6 - len(state.players[your_index].prize)`
    `Opp_Prize_Taken = 6 - len(state.players[1 - your_index].prize)`
    *Bobot: Sangat Tinggi (misal: 0.1 per selisih kartu)*
2.  **Board HP Advantage:** Total darah Pokemon aktif dan Bench.
    *Bobot: Sangat Rendah (misal: 0.0001 per HP).* Supaya AI tidak menghindari pertarungan berisiko tinggi yang menguntungkan secara Prize Card.

**Contoh Pseudo-code Potensial:**
```python
def calc_potential(state, your_index):
    # Hitung selisih prize cards (rentang -6 sampai 6)
    my_prize_taken = 6 - len(state.players[your_index].prize)
    opp_prize_taken = 6 - len(state.players[1 - your_index].prize)
    prize_diff = my_prize_taken - opp_prize_taken
    
    # Hitung selisih total HP di papan
    my_hp = sum([p.hp for p in state.players[your_index].active if p]) + \
            sum([p.hp for p in state.players[your_index].bench])
    opp_hp = sum([p.hp for p in state.players[1 - your_index].active if p]) + \
             sum([p.hp for p in state.players[1 - your_index].bench])
    hp_diff = my_hp - opp_hp
    
    # Hitung selisih jumlah kartu di deck (mencegah Deck Out & strategi Mill)
    my_deck_count = state.players[your_index].deckCount
    opp_deck_count = state.players[1 - your_index].deckCount
    deck_diff = my_deck_count - opp_deck_count
    
    # Formula Potensi (Maksimal nilai Prize sekitar +/- 0.6, tidak melebihi nilai Win/Loss)
    potential = (prize_diff * 0.1) + (hp_diff * 0.0001) + (deck_diff * 0.001)
    return potential
```

**Pada Training Loop / Gym Env step():**
```python
new_potential = calc_potential(new_state, your_index)
step_reward = new_potential - old_potential
old_potential = new_potential
```

## 3. Strict Time/Action Penalty (Mencegah Stalling)
Untuk memaksa agen mencari jalan tercepat menuju kemenangan tanpa bertele-tele:
*   **Time Penalty:** Berikan reward negatif yang sangat konstan namun kecil setiap step, contohnya `-0.001` per aksi (atau per giliran / `turnActionCount`).
Penalti ini memastikan AI tidak melakukan "pass" tanpa batas atau menukar opsi yang tidak perlu berkali-kali.

## 4. Masking untuk Invalid Actions, Bukan Penalti
Sering kali *developer* mencoba mencegah error dengan memberi `-1.0` reward ketika AI memilih langkah *illegal*. Ini adalah *bad practice* yang mengakibatkan **noise pada gradient** dan melambatkan training.
**Solusi:** Karena Kaggle/Engine menyediakan `select.option` (daftar *legal actions*), kita menggunakan **Action Masking**.
Set semua aksi yang tidak sah menjadi logit `-infinity` (misal `-1e9`) SEBELUM diteruskan ke layar *Softmax*. Dengan ini probabilitas AI memilih langkah illegal menjadi persis 0% tanpa harus merusak fungsi reward.

## Kesimpulan Total Formula Reward AI
Pada setiap pemanggilan aksi / *search_step*:
```python
# 1. Base Step
R_step = -0.001 # penalty efisiensi waktu

# 2. Shaping (Potential)
R_shaping = current_potential - previous_potential

# 3. Terminal State Result
R_terminal = 0.0
if state.result != -1:
    if state.result == your_index:
        R_terminal = +1.0
    elif state.result == 2: # Draw
        R_terminal = -0.1   # Anggap draw merugikan
    else:
        R_terminal = -1.0
        
# 4. Total Reward
Total_Reward = R_step + R_shaping + R_terminal
```
Sistem ini dipastikan bebas celah eksploitasi, konvergen secara terarah pada tujuan menang permainan (winning condition), dan akan menghasilkan agen dengan performa tangguh.

---

## 5. Integrasi Arsitektur: JAX GPU vs C++ Stateful Engine (CPU)

Penting untuk dicatat bahwa *engine* simulator TCG ini beroperasi dalam bahasa C++ yang diakses melalui `ctypes` pada `api.py`. Simulator C++ ini sangat **stateful** (menyimpan data status permainan dalam memori pointer seperti `agent_ptr`).

Sifat stateful ini **sangat bertentangan** dengan model JAX / Flax yang mensyaratkan fungsi-fungsi murni (pure functions) yang tanpa *side-effects* agar dapat dikompilasi oleh XLA (`jax.jit`) dan didistribusikan secara massal (`jax.vmap`). Jika pemanggilan API C++ dicoba dimasukkan ke dalam blok `jax.jit`, compiler JAX akan crash karena mendeteksi akses memori eksternal.

### Solusi Standar Industri: Asynchronous Vectorized Environments (Actor-Learner)
Untuk mengatasinya, sistem pelatihan tidak boleh mengeksekusi *environment* dari dalam model JAX. Kita harus memisahkan *Engine* dan *Brain* menggunakan pola Actor-Learner:

1. **CPU Workers (Actor/Lingkungan):**
   Gunakan **Python Multiprocessing** (misalnya via `gymnasium.vector.AsyncVectorEnv` atau Ray) untuk memijahkan (spawn) $N$ proses independen. Setiap proses/worker memuat instansi mandiri dari C++ DLL/SO dan menjalankan status game. Worker ini bertugas memanggil `api.py` dan mengekstrak *JSON Observation* menjadi **NumPy Arrays** statis.
2. **GPU Main Process (Learner/Brain):**
   Proses utama bertugas secara eksklusif mengatur tensor dan memori GPU. Alih-alih mengevaluasi state satu per satu, proses ini menunggu sekumpulan $N$ state dari CPU workers, menyatukannya (stack) menjadi *Batch* `(N, Features)`, lalu menjalankannya ke dalam fungsi JAX yang telah di-JIT:
   ```python
   @jax.jit
   def get_batched_actions(params, batch_obs, batch_mask, key):
       logits = model.apply(params, batch_obs)
       masked_logits = logits + ((1.0 - batch_mask) * -1e9)
       # ... sample action logic ...
       return actions
   ```
   Kumpulan hasil `actions` berwujud NumPy array ini lalu dikirim kembali secara terdistribusi ke masing-masing $N$ CPU workers.
3. **PPO Rollout Buffer:**
   Worker CPU merekam jejak langkah `(state, action, reward, next_state, log_prob)` ke dalam *Rollout Buffer*. Ketika buffer mencapai limit ukuran batch, keseluruhan buffer diserahkan ke JAX GPU untuk dieksekusi pembaruan gradiennya (PPO update) dalam satu gebrakan komputasi berkecepatan tinggi.

Pendekatan ini mengisolasi proses C++ yang 'kotor' dan tersendat di ranah CPU, sementara memastikan GPU bebas melakukan operasi tensor murni dengan performa XLA yang blazingly fast. Pastikan modul C++ dimuat ulang secara independen di setiap *subprocess* multiprocessing untuk menghindari *segmentation fault* atau bentrok pointer global.
