import os
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_episode(ep_id, save_dir):
    """
    Mengunduh satu episode dari Kaggle API menggunakan HTTP POST request.
    """
    url = "https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay"
    payload = {"episodeId": int(ep_id)}
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    save_path = os.path.join(save_dir, f"{ep_id}.json")
    
    # Skip jika file sudah ada (Resume support)
    if os.path.exists(save_path):
        return ep_id, True, "Already exists"
        
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            
            # Mendukung berbagai format respon dari Kaggle API
            replay_data = data.get("replay")
            if replay_data is None and "result" in data:
                replay_data = data["result"].get("replay")
                
            if replay_data:
                # Kaggle kadang mengirim JSON dalam bentuk string (stringify) atau langsung dictionary
                if isinstance(replay_data, str):
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(replay_data)
                else:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(replay_data, f)
                return ep_id, True, "Success"
            else:
                return ep_id, False, "No replay data in response"
        else:
            return ep_id, False, f"HTTP Error {res.status_code}"
            
    except Exception as e:
        return ep_id, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Kaggle Episode Replay Downloader (Multithreaded)")
    parser.add_argument("--input", type=str, required=True, help="Path ke file txt yang berisi daftar Episode ID (satu ID per baris).")
    parser.add_argument("--output", type=str, default="scraper/replays", help="Folder tujuan penyimpanan file JSON.")
    parser.add_argument("--workers", type=int, default=20, help="Jumlah thread (kecepatan download). Maksimal disarankan 50.")
    args = parser.parse_args()

    # Buat direktori output jika belum ada
    os.makedirs(args.output, exist_ok=True)
    
    # Baca daftar ID dari file
    episode_ids = []
    if os.path.exists(args.input):
        with open(args.input, "r") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    episode_ids.append(int(line))
    else:
        print(f"❌ Error: File input '{args.input}' tidak ditemukan.")
        return

    total = len(episode_ids)
    print(f"[*] Ditemukan {total} Episode ID untuk diunduh.")
    print(f"[*] Menggunakan {args.workers} threads paralel.")
    print(f"[*] Menyimpan ke folder: {args.output}\n")
    
    success_count = 0
    fail_count = 0
    
    # Multithreading eksekusi
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Peta tugas ke executor
        futures = {executor.submit(download_episode, ep_id, args.output): ep_id for ep_id in episode_ids}
        
        # Tampilkan progress bar
        with tqdm(total=total, desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                ep_id, success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    tqdm.write(f"[FAIL] ID {ep_id}: {msg}")
                pbar.update(1)
                
    elapsed = time.time() - start_time
    print(f"\n✅ Selesai dalam {elapsed:.2f} detik!")
    print(f"📊 Sukses: {success_count} | Gagal: {fail_count}")

if __name__ == "__main__":
    main()
