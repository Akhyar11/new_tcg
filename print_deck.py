import csv

def print_deck(filepath):
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
            
    print(f"Deck: {filepath}")
    for cid in sorted(deck):
        print(f"  - ID: {cid}, Name: {cards[cid]['Card Name']}, Category: {cards[cid]['Stage (Pokémon)/Type (Energy and Trainer)']}")

print_deck("agent_rl/deck/gen_deck_001.csv")
