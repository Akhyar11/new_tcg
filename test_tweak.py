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

original_deck = load_deck("agent_rl/deck/gen_deck_001.csv")
unique_cards = list(set(original_deck))

print("Testing replacements...")
for test_card in unique_cards:
    if test_card == 3: # Basic W Energy
        continue
    
    # Replace all instances of test_card with 1
    test_deck = [1 if c == test_card else c for c in original_deck]
    obs, _ = battle_start(test_deck, test_deck)
    if obs is not None:
        print(f"SUCCESS when replacing card ID: {test_card}")
    
# What if it's multiple cards? Let's just remove ALL Trainers and test
no_trainers_deck = [1 if c > 1000 else c for c in original_deck]
obs, _ = battle_start(no_trainers_deck, no_trainers_deck)
if obs is not None:
    print("SUCCESS when replacing all trainers!")
else:
    print("FAILED even without trainers.")
