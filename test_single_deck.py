import sys
from cg.game import battle_start

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
    return deck

deck = load_deck("agent_rl/deck/gen_deck_704.csv")
try:
    print(f"Deck length: {len(deck)}")
    obs_dict, _ = battle_start(deck, deck)
    print("Success!")
except Exception as e:
    print(f"Exception: {e}")
