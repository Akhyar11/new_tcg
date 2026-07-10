# Panduan Lanjutan: Dari Random Bot Menuju Self-Play

Dokumen ini berisi instruksi langkah-demi-langkah tentang apa yang harus dilakukan setelah seluruh arsitektur JAX PPO untuk Pokemon TCG ini selesai dibangun.

## 1. Memulai Pelatihan Massal (Fase 1)
Saat ini, model sudah siap untuk dilatih melawan **Random Bot**.
1. Buka file `agent_rl/train.py`.
2. Ubah baris konfigurasi total waktu eksperimen ke skala yang sangat besar. Ubah `TOTAL_TIMESTEPS` kembali menjadi `1000000` (atau lebih, tergantung kemampuan VRAM GPU/CPU Anda).
3. Jalankan perintah di terminal dengan memposisikan ROOT (*PYTHONPATH*) di direktori utama:
   ```bash
   PYTHONPATH=. python agent_rl/train.py
   ```
4. **Pantau Terminal:** Biarkan program berjalan (bisa memakan waktu berjam-jam). Anda harus fokus memantau metrik **Win Rate** dan **Avg Reward**. Awalnya *win rate* akan berada di sekitar 0%, namun perlahan-lahan AI akan mengerti pola dan meraih *win rate* yang terus menanjak.

---

## 2. Pemicu Transisi ke Self-Play (Win Rate > 90%)
**Jangan biarkan AI melawan Random Bot selamanya!**
Jika AI terus berlatih dengan musuh yang bodoh (acak) setelah ia fasih, AI akan terkena *overfitting* — ia hanya tahu taktik murahan untuk menang melawan bot acak, tetapi hancur jika melawan manusia sungguhan yang punya strategi.

**Kapan Waktu yang Tepat?**
Ketika log terminal Anda secara konsisten menunjukkan angka:
`Win Rate: 90.0%` hingga `95.0%`

**Apa yang Harus Anda Lakukan Setelah Angka Ini Tercapai?**
Hentikan *training* sementara (Tekan `Ctrl+C`). Karena sistem kita secara otomatis melakukan autosave ke `checkpoints/model_final.msgpack` dan setiap kelipatan 50, memori AI akan tetap aman.

Panggil kembali AI Assistant Anda dan berikan pesan ini:
> *"Agen sudah bisa mengalahkan bot acak dengan win-rate di atas 90%. Ayo kita modifikasi kodenya menjadi mode Self-Play!"*

### Apa yang Akan Kita Lakukan di Mode Self-Play Nanti?
1. **Memodifikasi `vector_env.py`:** Kita akan menghapus fungsi `advance_to_player0` yang menggerakkan lawan secara acak. Sebaliknya, saat giliran Player 1 tiba, *environment* akan membeku dan meminta tebakan langkah (*action*) dari Jaringan Saraf JAX, persis seperti yang dilakukannya pada Player 0.
2. **Memodifikasi `train.py`:** Kita akan memuat (*load*) memori dari `checkpoints/model_final.msgpack` (yang sudah pintar melawan random bot) sebagai bobot otak untuk Player 1.
3. Dengan begini, AI Player 0 (yang sedang belajar) akan dipaksa bertarung mati-matian melawan AI Player 1 (dirinya sendiri dari masa lalu). Hal ini akan memicu *arms-race* (perlombaan senjata) kecerdasan di mana ia dipaksa menciptakan taktik-taktik cerdas yang tersembunyi.

---

## 3. Pengembangan Tambahan (Opsional)
Sembari menunggu AI berlatih, Anda juga bisa menugaskan asisten untuk membuat:
*   **Visualizer / Play-vs-Bot:** Sebuah skrip untuk melawan bot buatan Anda sendiri menggunakan *Terminal/CLI*.
*   **Grafik Evaluasi:** Mengekstrak metrik dari *logs* untuk divisualisasikan dengan matplotlib.
