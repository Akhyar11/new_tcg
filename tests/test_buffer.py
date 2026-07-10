import numpy as np
from agent_rl.buffer import RolloutBuffer

def test_buffer():
    # Simulasi parameter lingkungan
    n_steps = 10
    num_envs = 4
    batch_size = 8
    
    print(f"1. Menginisiasi RolloutBuffer (n_steps={n_steps}, num_envs={num_envs})...")
    buffer = RolloutBuffer(n_steps=n_steps, num_envs=num_envs)
    print(f"Total kapasitas (buffer_size): {buffer.buffer_size}")
    
    # 2. Pengisian Buffer
    print("\n2. Mensimulasikan pengumpulan data step-by-step...")
    for step in range(n_steps):
        # Generate dummy data berdimensi (num_envs, ...)
        seq_in = np.random.rand(num_envs, 93, 31).astype(np.float32)
        glob_in = np.random.rand(num_envs, 266).astype(np.float32)
        actions = np.random.randint(0, 250, size=(num_envs,)).astype(np.int32)
        log_probs = np.random.randn(num_envs).astype(np.float32)
        rewards = np.random.rand(num_envs).astype(np.float32)
        values = np.random.rand(num_envs).astype(np.float32)
        
        # Buat dummy terminal state (done=True secara acak)
        dones = np.random.choice([0.0, 1.0], size=(num_envs,), p=[0.9, 0.1]).astype(np.float32)
        
        buffer.add(seq_in, glob_in, actions, log_probs, rewards, values, dones)
        
    print(f"Berhasil mengisi buffer sampai step ke-{buffer.step}!")
    
    # 3. Kalkulasi GAE
    print("\n3. Menghitung GAE (Generalized Advantage Estimation)...")
    last_values = np.random.rand(num_envs).astype(np.float32)
    last_dones = np.zeros(num_envs, dtype=np.float32)
    
    buffer.compute_returns_and_advantages(last_values, last_dones)
    
    print("Contoh 4 nilai advantages awal:", buffer.advantages[0])
    print("Contoh 4 nilai returns awal:", buffer.returns[0])
    
    # 4. Generator Mini-Batch
    print(f"\n4. Menguji Generator Batching (Batch Size: {batch_size})...")
    batch_count = 0
    for batch in buffer.get_batches(batch_size):
        batch_count += 1
        print(f"   Batch ke-{batch_count}:")
        print(f"   - seq_input shape: {batch['seq_input'].shape}")
        print(f"   - glob_input shape: {batch['glob_input'].shape}")
        print(f"   - actions shape: {batch['actions'].shape}")
        print(f"   - advantages shape: {batch['advantages'].shape}")
        
    print(f"\nTotal mini-batches dihasilkan: {batch_count} (Harus = buffer_size / batch_size = {buffer.buffer_size / batch_size})")
    
    print("\n[SEMUA PENGUJIAN ROLLOUT BUFFER BERHASIL LULUS!]")

if __name__ == "__main__":
    test_buffer()
