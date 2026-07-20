import os
import sys

# Setup CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Add ROOT to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from tcg_core.agents import FFAgent, LSTMAgent

# Import models
from tcg_core.models.ff import PokemonAgent as FFModel
from tcg_core.models.lstm import PokemonAgent as LSTMModel

# Import unified action mapping
import tcg_core.action_mapping as action_mapping

from tcg_core.environment import TCGEnvironment

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                deck.append(int(line))
    if len(deck) != 60: return None
    return deck

def simulate_game(agent_p0, agent_p1, deck0, deck1):
    agent_p0.reset()
    agent_p1.reset()
    
    env = TCGEnvironment()
    obs, done = env.reset(deck0, deck1)
    
    step_count = 0
    while not done and step_count <= 250:
        step_count += 1
        your_idx = obs.current.yourIndex
        
        if your_idx == 0:
            choices = agent_p0.select_action(obs)
        else:
            choices = agent_p1.select_action(obs)
            
        obs, _, done, info = env.step(choices)
                
    result = info.get("result", -1) if done else -1
    env.close()
    return result

def test():
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    path_ff = os.path.join(checkpoints_dir, "model_final.msgpack")
    path_lstm = os.path.join(checkpoints_dir, "model_lstm_final.msgpack")

    print("Initializing OOP Agents...")
    agent_ff = FFAgent("FF", FFModel, action_mapping, path_ff)
    agent_lstm = LSTMAgent("LSTM", LSTMModel, action_mapping, path_lstm)
    print("Agents Initialized successfully.")

    deck_path = os.path.join(ROOT, "new_deck", "Mega Lucario Aura Strike.csv")
    deck0 = load_deck(deck_path)
    deck1 = load_deck(deck_path)

    print("Running OOP Test Game 1 (P0: LSTM, P1: FF)...")
    res1 = simulate_game(agent_lstm, agent_ff, deck0, deck1)
    print(f"Result Game 1: Winner = {'P0 (LSTM)' if res1 == 0 else 'P1 (FF)' if res1 == 1 else 'Tie'}")

    print("Running OOP Test Game 2 (P0: FF, P1: LSTM)...")
    res2 = simulate_game(agent_ff, agent_lstm, deck0, deck1)
    print(f"Result Game 2: Winner = {'P0 (FF)' if res2 == 0 else 'P1 (LSTM)' if res2 == 1 else 'Tie'}")

if __name__ == "__main__":
    test()
