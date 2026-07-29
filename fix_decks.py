import os
import csv
from collections import Counter

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, 'cg', 'database.csv')
    deck_dir = os.path.join(root_dir, 'new_deck')
    
    # Energy mapping
    type_to_energy_id = {
        '{G}': 1,
        '{R}': 2,
        '{W}': 3,
        '{L}': 4,
        '{P}': 5,
        '{F}': 6,
        '{D}': 7,
        '{M}': 8,
    }
    
    # Load DB
    pokemon_types = {}
    with open(db_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if len(row) > 10:
                card_id = int(row[0])
                card_category = row[4] # Stage (Pokémon)
                card_type = row[9] # Type
                if 'Pokémon' in card_category:
                    pokemon_types[card_id] = card_type

    # Process decks in multiple directories
    deck_dirs = ['new_deck', 'deck_generated', 'deck_ga']
    for d in deck_dirs:
        d_path = os.path.join(root_dir, d)
        if not os.path.exists(d_path): continue
        for filename in os.listdir(d_path):
            if not filename.endswith('.csv'): continue
            filepath = os.path.join(d_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        
        deck = []
        for line in lines:
            if line.strip().isdigit():
                deck.append(int(line.strip()))
                
        if not deck:
            continue
            
        # Analyze Pokemon types in deck
        types_in_deck = []
        basic_energies_in_deck = []
        other_cards = []
        for card_id in deck:
            if card_id in pokemon_types:
                ptype = pokemon_types[card_id]
                if ptype in type_to_energy_id: # Ignore Colorless {C} or multiple types if any
                    types_in_deck.append(ptype)
                other_cards.append(card_id)
            elif 1 <= card_id <= 8:
                basic_energies_in_deck.append(card_id)
            else:
                other_cards.append(card_id)
                
        if not types_in_deck or not basic_energies_in_deck:
            continue
            
        most_common_type = Counter(types_in_deck).most_common(1)[0][0]
        correct_energy_id = type_to_energy_id[most_common_type]
        
        # Replace basic energies
        replaced = False
        new_deck = []
        for card_id in deck:
            if 1 <= card_id <= 8:
                if card_id != correct_energy_id:
                    replaced = True
                new_deck.append(correct_energy_id)
            else:
                new_deck.append(card_id)
                
        if replaced:
            print(f"Fixed {filename}: changed energies to {most_common_type} (ID {correct_energy_id})")
            with open(filepath, 'w', encoding='utf-8') as f:
                for card_id in new_deck:
                    f.write(f"{card_id}\n")

if __name__ == '__main__':
    main()
