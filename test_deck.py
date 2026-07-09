from cg.game import battle_start
import sys

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
    return deck

try:
    deck = load_deck("agent_rl/deck/gen_deck_237.csv")
    print(f"Deck length: {len(deck)}")
    obs_dict, _ = battle_start(deck, deck)
    print("Battle started successfully!")
except Exception as e:
    print(f"Failed: {e}")
