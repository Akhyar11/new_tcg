import csv
from cg.game import battle_start, battle_finish

csv_path = "agent_rl/EN_Card_Data.csv"
basics = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if 'Basic Pokémon' in row['Stage (Pokémon)/Type (Energy and Trainer)']:
            basics.append(int(row['Card ID']))

valid_basics = []
invalid_basics = []

print(f"Testing {len(basics)} basics...")

# Use Energy for the rest of the deck (Card ID 1 is Basic {G} Energy)
for cid in basics:
    deck = [cid]*4 + [1]*56
    try:
        battle_finish()
        obs_dict, _ = battle_start(deck, deck)
        if obs_dict is not None and 'current' in obs_dict and obs_dict['current'] is not None:
            valid_basics.append(cid)
        else:
            invalid_basics.append(cid)
    except:
        invalid_basics.append(cid)

print(f"Valid basics: {len(valid_basics)}, Invalid basics: {len(invalid_basics)}")
if len(invalid_basics) > 0:
    print(f"First 10 invalid basics: {invalid_basics[:10]}")
