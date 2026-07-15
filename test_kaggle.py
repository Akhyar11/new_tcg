import os
import sys
import tempfile
from unittest.mock import MagicMock

# Mock ML libraries so this script can be imported and run anywhere instantly
sys.modules['jax'] = MagicMock()
sys.modules['jax.numpy'] = MagicMock()
sys.modules['optax'] = MagicMock()
sys.modules['flax'] = MagicMock()
sys.modules['flax.serialization'] = MagicMock()
sys.modules['flax.jax_utils'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['agent_rl.model'] = MagicMock()
sys.modules['agent_rl.vector_env'] = MagicMock()
sys.modules['agent_rl.buffer'] = MagicMock()
sys.modules['agent_rl.ppo_update'] = MagicMock()

# Import the actual functions from our training module
from agent_rl.train import upload_to_kaggle, download_from_kaggle

def run_test():
    print("=== KAGGLE API CONNECTION TEST ===")
    print("Testing download and upload flows using your credentials...\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("--- PHASE 1: TESTING DOWNLOAD ---")
        try:
            download_from_kaggle(tmpdir)
            files = os.listdir(tmpdir)
            print(f"[SUCCESS] Download completed. Files found: {files}")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return False
            
        if "model_final.msgpack" in files:
            print("\n--- PHASE 2: TESTING UPLOAD ---")
            print("Uploading the exact downloaded files to test upload access safely (no corruption)...")
            try:
                upload_to_kaggle(tmpdir, message="Kaggle VM connection test")
                print("[SUCCESS] Upload and version finalization completed successfully!")
                return True
            except Exception as e:
                print(f"[ERROR] Upload failed: {e}")
                return False
        else:
            print("[ERROR] Downloaded files did not contain model_final.msgpack.")
            return False

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n🎉 SUCCESS: Your Kaggle VM can download and upload successfully with write-access!")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Connection test failed. Check the errors above.")
        sys.exit(1)
