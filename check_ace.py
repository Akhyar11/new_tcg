import csv
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

csv_path = "agent_rl/EN_Card_Data.csv"
cards = {}
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cards[int(row['Card ID'])] = row

for cid in original_deck:
    if 'ACE SPEC' in cards[cid]['Rule'] or 'ACE SPEC' in cards[cid]['Stage (Pokémon)/Type (Energy and Trainer)']:
        print(f"Deck contains ACE SPEC: {cards[cid]['Card Name']} (ID: {cid})")
