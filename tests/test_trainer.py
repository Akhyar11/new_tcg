import os
import sys

# Setup CPU for small test
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from tcg_core.agents import LSTMAgent
from tcg_core.trainer import TrainerPPO
from tcg_core.models.lstm import PokemonAgent as LSTMModel
import tcg_core.action_mapping as action_mapping

def test():
    print("Initializing OOP LSTMAgents...")
    agent_p0 = LSTMAgent("LSTM_P0", LSTMModel, action_mapping)
    agent_p1 = LSTMAgent("LSTM_P1", LSTMModel, action_mapping)
    
    config = {
        "num_envs": 2,          # Very small for quick test
        "n_steps": 32,          # Minimum sequence length
        "batch_size": 64,       # (2 * 32 = 64)
        "new_deck_path": os.path.join(ROOT, "new_deck"),
        "gen_deck_path": os.path.join(ROOT, "deck_generated"),
        "save_dir": os.path.join(ROOT, "tcg_core_test_checkpoints")
    }

    print("Initializing TrainerPPO...")
    trainer = TrainerPPO(agent_p0, agent_p1, config)
    
    print("Running TrainerPPO for 128 timesteps (2 updates)...")
    try:
        trainer.train(total_timesteps=128)
        print("TrainerPPO test passed successfully!")
    except Exception as e:
        print(f"TrainerPPO test failed with exception: {e}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    test()
