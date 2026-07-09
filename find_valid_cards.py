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

deck_dir = "agent_rl/deck"
files = [f for f in os.listdir(deck_dir) if f.endswith(".csv")]

successful_cards = set()
failed_cards = set()

for file in files:
    filepath = os.path.join(deck_dir, file)
    deck = load_deck(filepath)
    try:
        battle_finish()
        obs_dict, _ = battle_start(deck, deck)
        if obs_dict is not None and 'current' in obs_dict and obs_dict['current'] is not None:
            successful_cards.update(deck)
        else:
            failed_cards.update(deck)
    except:
        failed_cards.update(deck)

# Cards that are only in failed decks are likely invalid
invalid_cards = failed_cards - successful_cards
print(f"Number of valid cards found (appear in at least one successful deck): {len(successful_cards)}")
print(f"Number of likely invalid cards: {len(invalid_cards)}")

valid_cards_list = sorted(list(successful_cards))
with open('agent_rl/valid_cards.txt', 'w') as f:
    for c in valid_cards_list:
        f.write(f"{c}\n")
