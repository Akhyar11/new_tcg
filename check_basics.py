import csv

def check_basics(filepath):
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
            
    num_basics = sum(1 for cid in deck if 'Basic Pokémon' in cards[cid]['Stage (Pokémon)/Type (Energy and Trainer)'])
    print(f"Deck has {num_basics} Basic Pokemon.")
    for cid in deck:
        if 'Basic Pokémon' in cards[cid]['Stage (Pokémon)/Type (Energy and Trainer)']:
            print(f"  - {cards[cid]['Card Name']} (ID: {cid})")

check_basics("agent_rl/deck/gen_deck_001.csv")
