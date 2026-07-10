# Rencana Pembuatan GUI & Mode Bermain Manusia vs AI

Dokumen ini memuat tahapan terstruktur (tasks) untuk membangun antarmuka grafis (GUI) interaktif yang memungkinkan pemain manusia bertanding langsung melawan agen AI Pokémon TCG (JAX/Flax) Anda.

## 1. Pemilihan Arsitektur & Teknologi GUI
Mengingat game ini memiliki kompleksitas visual yang tinggi (kartu, tumpukan deck, efek serangan), pendekatan **Aplikasi Web (Web-based GUI)** adalah yang paling direkomendasikan karena memberikan fleksibilitas desain modern dan animasi yang mulus.

*   **Backend:** Python (FastAPI).
    *   Berfungsi sebagai jembatan yang menjalankan C++ Engine (`cg`) dan mengelola status permainan.
    *   Mengatur model JAX (`model.py`) untuk giliran AI.
    *   Komunikasi waktu nyata dengan Frontend menggunakan **WebSockets**.
*   **Frontend:** Vanilla JS / React (dengan CSS Modern).
    *   Membuat desain antarmuka yang sangat premium (animasi hover kartu, efek *glassmorphism*, dark mode).
*   *(Alternatif jika ingin murni Desktop)*: Pygame atau PyQt6. Namun tampilan UI-nya akan kalah estetis dibanding Web.

## 2. Persiapan Aset (Game Assets)
Sebelum membuat UI, kita perlu mengumpulkan aset visual yang berkualitas.
*   **[ ] Scraping/Download Gambar Kartu:**
    *   Menarik gambar kartu berdasarkan `card_id` menggunakan layanan seperti Pokémon TCG API (https://pokemontcg.io/).
    *   Menyimpan aset gambar di folder lokal (misal: `assets/cards/`).
*   **[ ] Aset UI Tambahan:**
    *   Desain bagian belakang kartu (Card Back).
    *   Ikon Energi (Grass, Fire, Water, dll.) untuk menempelkan status energi di antarmuka.
    *   Gambar Playmat/Board (Latar belakang permainan).
    *   Token Damage, Penanda Racun/Terbakar (Poison/Burn markers), dan penanda VSTAR.

## 3. Tahapan Implementasi (Milestones)

### Fase 1: Integrasi Backend API (Game Server)
*   [ ] Membuat skrip server (contoh: `server.py`) menggunakan FastAPI.
*   [ ] Membuat *Session Manager* agar server bisa menginisiasi C++ game dari `cg.game.battle_start()`.
*   [ ] Membuat endpoint WebSocket untuk mengirim **Observation (Game State)** ke browser.
*   [ ] Menyiapkan fungsi khusus yang menerjemahkan klik pemain (di browser) menjadi `Action Index` (0-249) yang dimengerti oleh C++ engine.

### Fase 2: Pembangunan Antarmuka (Frontend Board)
*   [ ] Membangun layout Playmat (Zona Active, Bench, Hand, Discard, Prize, Lost Zone, Stadium).
*   [ ] Mengimplementasikan *render* gambar kartu yang sesuai dengan ID dari data JSON server.
*   [ ] Menambahkan animasi klik dan seret (*drag-and-drop* atau *click-to-select*) untuk kemudahan bermain.
*   [ ] Membangun menu *Prompt / Select Option* (karena game sering meminta interaksi spesifik seperti memilih kartu dari deck).

### Fase 3: Penggabungan AI (AI Integration)
*   [ ] Memuat bobot checkpoint JAX (`model_update_XXX.msgpack`) di backend FastAPI.
*   [ ] Memastikan saat giliran musuh tiba, server otomatis membungkus *observation*, memprosesnya lewat AI, dan mengaplikasikan aksi AI ke C++ Engine.
*   [ ] Memberikan jeda waktu (sekitar 1-2 detik) saat AI memikirkan langkahnya, dan memberi indikator "AI sedang berpikir..." di layar agar terasa nyata.

### Fase 4: Poles Estetika (Polish & FX)
*   [ ] Animasi saat kartu diserang (guncangan layar / efek partikel).
*   [ ] Suara efek (SFX) sederhana saat menarik kartu atau mengeluarkan energi.
*   [ ] Desain papan pemenang (Victory/Defeat screen).

## Kesimpulan
Pendekatan ini memisahkan logika berat (C++ dan AI JAX) di latar belakang, sementara memberikan pemain pengalaman bermain yang menawan di browser. Langkah pertama yang harus kita ambil adalah **mengumpulkan aset gambar kartu (Scraping)** dan **membangun kerangka FastAPI**.
