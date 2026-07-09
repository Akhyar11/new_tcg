import os
from cg.game import battle_start, battle_finish

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
    return deck

deck_dir = "agent_rl/deck_generated"
files = [f for f in os.listdir(deck_dir) if f.endswith(".csv")]

failures = 0
total = len(files)

print(f"Testing {total} decks...")

for file in files:
    filepath = os.path.join(deck_dir, file)
    deck = load_deck(filepath)
    try:
        battle_finish() # clear any existing battle
        obs_dict, _ = battle_start(deck, deck)
        if obs_dict is None:
            print(f"{file}: obs_dict is None")
            failures += 1
        elif 'current' not in obs_dict:
            print(f"{file}: 'current' not in obs_dict")
            failures += 1
        elif obs_dict['current'] is None:
            print(f"{file}: 'current' is None")
            failures += 1
    except Exception as e:
        print(f"{file}: Exception {e}")
        failures += 1

print(f"Total failures: {failures} out of {total}")
