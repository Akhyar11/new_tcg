import csv
from collections import Counter

def check_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
                
    csv_path = "agent_rl/EN_Card_Data.csv"
    cards = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards[int(row['Card ID'])] = row
            
    names = [cards[cid]['Card Name'] for cid in deck]
    counts = Counter(names)
    
    violators = {name: count for name, count in counts.items() if count > 4 and 'Basic Energy' not in cards[deck[names.index(name)]]['Stage (Pokémon)/Type (Energy and Trainer)']}
    
    return len(deck), violators

print(check_deck("agent_rl/deck/gen_deck_001.csv"))
