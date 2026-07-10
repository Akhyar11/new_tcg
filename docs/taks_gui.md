# Master Plan: Pokémon TCG Web Application

Dokumen ini berisi struktur pengembangan untuk versi penuh dari Pokémon TCG Web Game.

## 1. Tech Stack
*   **Frontend**: Next.js (App Router), TypeScript, Vanilla CSS (Premium & Modern Aesthetics).
*   **Backend Server (Game API & WebSocket)**: Python FastAPI (terhubung ke C++ TCG Engine dan model AI JAX).
*   **Database**: MySQL (untuk menyimpan data *user*, *deck*, dan *friend list*).
*   **State Management & ORM**: Prisma ORM (untuk interaksi Next.js ke MySQL) atau koneksi langsung dari FastAPI (SQLAlchemy).

## 2. Fitur Utama (Core Features)

### A. Deck Builder
*   **Database Kartu**: Menampilkan galeri semua 1.200+ kartu yang tersedia.
*   **Drag & Drop UI**: Memasukkan dan mengeluarkan kartu dari deck.
*   **Validasi**: Pengecekan otomatis aturan deck (harus 60 kartu, batas maksimal kartu ber-nama sama, dll).
*   **Penyimpanan**: Menyimpan konfigurasi deck ke dalam MySQL agar bisa dipakai sewaktu-waktu.

### B. Bermain Melawan AI (PvE)
*   **Mode Praktik**: Bermain melawan agen AI JAX.
*   **Tingkat Kesulitan**: (Bisa menyesuaikan checkpoint model AI yang dimuat).
*   **Animasi Real-time**: Feedback visual saat AI bergerak (kartu terpasang, serangan dilancarkan).

### C. Bermain Multiplayer (PvP)
*   **Lobby / Matchmaking**: Sistem antrean (queue) pemain.
*   **Room System**: WebSocket mereplika status game untuk 2 klien (pemain 1 dan pemain 2) yang tersambung ke FastAPI *Game Session*.
*   **Keamanan**: Semua aturan (rules) dijalankan di *server (C++ Engine)* untuk mencegah curang (*anti-cheat*).

## 3. Fitur Lanjutan (Advanced)

### D. Sistem Teman & Jejaring (Social)
*   **Friend List**: Menambahkan pemain lain menggunakan Player ID atau Username (tersimpan di MySQL).
*   **Direct Challenge**: Mengirim undangan duel langsung ke teman yang sedang *online*.

## 4. Tahapan Pekerjaan Saat Ini
*   [x] Persiapan Aset (Ribuan kartu berhasil di-download).
*   [x] Inisialisasi Proyek Web (Next.js telah dibuat di folder `web_app`).
*   [ ] Konfigurasi Database (Menyiapkan skema MySQL untuk `users`, `decks`, `friends`).
*   [ ] Desain UI/UX (Membangun halaman beranda berdesain modern, gelap, dan premium di Next.js).
*   [ ] Pembuatan Halaman *Deck Builder*.
*   [ ] Integrasi Game Board dengan WebSocket FastAPI.
