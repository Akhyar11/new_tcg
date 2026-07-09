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
try:
    deck = load_deck(sys.argv[1])
    obs, _ = battle_start(deck, deck)
    if obs is None:
        print("obs is None")
    elif 'current' not in obs:
        print("no current in obs")
    elif obs['current'] is None:
        print("current is None")
except Exception as e:
    import traceback
    traceback.print_exc()
