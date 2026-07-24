import os
import sys

# Setup environment for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tcg_core.agents import LSTMAgent
from tcg_core.trainer import TrainerPPO
from tcg_core.kaggle_sync import upload_to_kaggle, download_from_kaggle
from tcg_core.models.ptr import PokemonAgent as PTRModel
import tcg_core.action_mapping as action_mapping

def main():
    print("=== TCG AI TRAINING (CURRICULUM: V2 vs V1 -> Self Play) ===")
    
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Download latest from Kaggle before starting
    print("Mendownload checkpoint terbaru dari Kaggle...")
    try:
        download_from_kaggle(save_dir)
    except Exception as e:
        print(f"Gagal mendownload dari Kaggle: {e}")

    path_ptr_v1 = os.path.join(save_dir, "model_lstm_pointer_final.msgpack")
    path_ptr_v2 = os.path.join(save_dir, "model_lstm_pointer_v2_final.msgpack")
    
    print("Initializing PTR Agents for Curriculum Learning...")
    # P0 adalah agen yang akan kita latih (V2). Kita muat checkpoint V2 saat ini.
    agent_p0 = LSTMAgent("PTR_V2_Trainee", PTRModel, action_mapping, 
                         checkpoint_path=path_ptr_v2 if os.path.exists(path_ptr_v2) else (path_ptr_v1 if os.path.exists(path_ptr_v1) else None))
    
    # P1 adalah lawan pertamanya (V1). Kita set dengan checkpoint V1 yang terbukti kuat.
    agent_p1 = LSTMAgent("PTR_V1_Opponent", PTRModel, action_mapping, 
                         checkpoint_path=path_ptr_v1 if os.path.exists(path_ptr_v1) else None)
    
    # Keterangan:
    # Karena class TrainerPPO di `tcg_core/trainer.py` sudah memiliki logika Self-Play otomatis (baris 270):
    # Ketika win rate P0 (V2) mencapai >= 57% berturut-turut melawan P1 (V1), 
    # maka TrainerPPO akan menimpa bobot P1 dengan P0. 
    # Mulai dari titik tersebut, proses training otomatis berubah menjadi V2 melawan V2 (Self-Play).
    
    config = {
        "num_envs": 16,
        "n_steps": 256,
        "batch_size": 512,
        "epochs": 1,
        "learning_rate": 5e-5,
        "entropy_coef": 0.05,
        "clip_ratio": 0.2,
        "new_deck_path": os.path.join(os.path.dirname(__file__), "new_deck"),
        "gen_deck_path": os.path.join(os.path.dirname(__file__), "deck_generated"),
        "save_dir": save_dir,
        "save_name_base": "model_lstm_pointer_v2_base.msgpack",
        "save_name_final": "model_lstm_pointer_v2_final.msgpack",
        "use_wandb": True
    }

    trainer = TrainerPPO(agent_p0, agent_p1, config)
    
    # Training Loop (30M Timesteps)
    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", 30000000))
    try:
        trainer.train(total_timesteps=total_timesteps, finetune_mode=False)
        print("Uploading final model to Kaggle...")
        upload_to_kaggle(save_dir, message="Final Training Curriculum Checkpoint")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saved current progress.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
