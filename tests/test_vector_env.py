import time
import numpy as np
from tcg_core.vector_env import VectorEnv

def test_vector_env():
    num_envs = 4
    print(f"Menginisiasi {num_envs} pekerja CPU secara paralel...")
    env = VectorEnv(num_envs=num_envs)
    
    try:
        print("Melakukan Reset serentak...")
        obs = env.reset()
        print(f"Bentuk BATCH seq_input: {obs['seq_input'].shape} (Harus: 4, 113, 31)")
        print(f"Bentuk BATCH glob_input: {obs['glob_input'].shape} (Harus: 4, 266)")
        
        print("\nMelakukan 10 step acak paralel...")
        start_time = time.time()
        
        for i in range(10):
            # Simulasi AI memprediksi aksi acak (0-249)
            random_actions = [np.random.randint(0, 250) for _ in range(num_envs)]
            
            # Step async
            obs, rewards, dones = env.step(random_actions)
            
            # Cek jika ada env yang selesai
            for j in range(num_envs):
                if dones[j]:
                    print(f"Worker {j} Game Over! Resetting...")
                    # Dalam RL sebenarnya kita mereset worker spesifik, tapi untuk test biarkan.
        
        end_time = time.time()
        print(f"\n10 Step paralel (total 40 operasi C++) selesai dalam {end_time - start_time:.4f} detik!")
        print(f"Contoh Reward dari 4 worker: {rewards}")
        
    finally:
        print("Menutup semua pekerja...")
        env.close()

if __name__ == "__main__":
    test_vector_env()
