import os
import sys

# Setup environment for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tcg_core.agents import LSTMAgent
from tcg_core.trainer import TrainerPPO
from tcg_core.kaggle_sync import upload_to_kaggle, download_from_kaggle
from tcg_core.models.lstm import PokemonAgent as LSTMModel
import tcg_core.action_mapping as action_mapping

def main():
    print("=== TCG AI TRAINING (OOP MODE) ===")
    
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Download latest from Kaggle before starting
    print("Mendownload checkpoint terbaru dari Kaggle...")
    try:
        download_from_kaggle(save_dir)
    except Exception as e:
        print(f"Gagal mendownload dari Kaggle: {e}")

    path_lstm_final = os.path.join(save_dir, "model_lstm_final.msgpack")
    
    # Initialize agents (P0 and P1 start with the same weights)
    print("Initializing Agents...")
    agent_p0 = LSTMAgent("LSTM_P0", LSTMModel, action_mapping, checkpoint_path=path_lstm_final if os.path.exists(path_lstm_final) else None)
    agent_p1 = LSTMAgent("LSTM_P1", LSTMModel, action_mapping, checkpoint_path=path_lstm_final if os.path.exists(path_lstm_final) else None)
    
    config = {
        "num_envs": 8,
        "n_steps": 128,
        "batch_size": 64,
        "epochs": 1,
        "learning_rate": 3e-4,
        "entropy_coef": 0.05,
        "clip_ratio": 0.2,
        "new_deck_path": os.path.join(os.path.dirname(__file__), "new_deck"),
        "gen_deck_path": os.path.join(os.path.dirname(__file__), "deck_generated"),
        "save_dir": save_dir,
        "save_name_base": "model_lstm_base.msgpack",
        "save_name_final": "model_lstm_final.msgpack"
    }

    trainer = TrainerPPO(agent_p0, agent_p1, config)
    
    # Training Loop (20M Timesteps)
    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", 20000000))
    try:
        trainer.train(total_timesteps=total_timesteps, finetune_mode=False)
        print("Uploading final model to Kaggle...")
        upload_to_kaggle(save_dir, message="Final Training OOP Checkpoint")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saved current progress.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
