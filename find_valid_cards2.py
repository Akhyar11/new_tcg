import os
import csv
from cg.game import battle_start, battle_finish

csv_path = "agent_rl/EN_Card_Data.csv"
all_cids = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_cids.append(int(row['Card ID']))
all_cids = list(set(all_cids))

valid_cards = []
invalid_cards = []

base_deck = [1] * 55 + [65]*4 # 55 Basic Green Energy + 4 Dunsparce (Basic Pokemon)

print(f"Testing {len(all_cids)} cards...")

for cid in all_cids:
    deck = base_deck + [cid]
    try:
        battle_finish()
        obs_dict, _ = battle_start(deck, deck)
        if obs_dict is not None and 'current' in obs_dict and obs_dict['current'] is not None:
            valid_cards.append(cid)
        else:
            invalid_cards.append(cid)
    except:
        invalid_cards.append(cid)

print(f"Total valid cards: {len(valid_cards)}")
print(f"Total invalid cards: {len(invalid_cards)}")

with open('agent_rl/valid_cards.txt', 'w') as f:
    for c in sorted(valid_cards):
        f.write(f"{c}\n")

if len(invalid_cards) > 0:
    print(f"First 10 invalid cards: {invalid_cards[:10]}")
