import csv
import random
import os

def generate_random_decks(csv_path, output_dir, num_decks=100):
    cards = {}
    pokemon_basics = []
    pokemon_stage1 = {}
    pokemon_stage2 = {}
    trainers = []
    energies = []

    # Read the card database
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['Card ID']
            cname = row['Card Name']
            stage = row['Stage (Pokémon)/Type (Energy and Trainer)']
            prev = row['Previous stage']
            ctype = row['Type']
            
            cards[cid] = row
            
            # Simple categorization based on the 'Stage/Type' column
            if 'Basic Pokémon' in stage:
                pokemon_basics.append(cid)
            elif 'Stage 1 Pokémon' in stage:
                if prev not in pokemon_stage1:
                    pokemon_stage1[prev] = []
                pokemon_stage1[prev].append(cid)
            elif 'Stage 2 Pokémon' in stage:
                if prev not in pokemon_stage2:
                    pokemon_stage2[prev] = []
                pokemon_stage2[prev].append(cid)
            elif 'Item' in stage or 'Supporter' in stage or 'Stadium' in stage or 'Pokémon Tool' in stage:
                trainers.append(cid)
            elif 'Basic Energy' in stage:
                energies.append(cid)

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loaded {len(pokemon_basics)} Basics, {len(trainers)} Trainers, {len(energies)} Energies.")

    for i in range(num_decks):
        deck = []
        name_counts = {}

        def can_add(cname):
            return name_counts.get(cname, 0) < 4

        def add_card(cid, count):
            cname = cards[cid]['Card Name']
            added = 0
            for _ in range(count):
                # Basic Energy can exceed 4 copies
                if len(deck) < 60 and (can_add(cname) or 'Basic Energy' in cards[cid]['Stage (Pokémon)/Type (Energy and Trainer)']):
                    deck.append(cid)
                    name_counts[cname] = name_counts.get(cname, 0) + 1
                    added += 1
            return added

        # 1. Pick 2-4 evolutionary lines (Pokemon)
        num_lines = random.randint(2, 4)
        for _ in range(num_lines):
            if not pokemon_basics: continue
            base_cid = random.choice(pokemon_basics)
            base_name = cards[base_cid]['Card Name']
            
            add_card(base_cid, random.randint(2, 4))
            
            # Stage 1
            if base_name in pokemon_stage1:
                s1_cid = random.choice(pokemon_stage1[base_name])
                s1_name = cards[s1_cid]['Card Name']
                add_card(s1_cid, random.randint(2, 3))
                
                # Stage 2
                if s1_name in pokemon_stage2:
                    s2_cid = random.choice(pokemon_stage2[s1_name])
                    add_card(s2_cid, random.randint(1, 2))

        # 2. Add Trainers (aim for ~35 trainers, so fill up to ~45-48 cards)
        target_size = random.randint(40, 48)
        max_attempts = 1000
        attempts = 0
        while len(deck) < target_size and attempts < max_attempts:
            attempts += 1
            if not trainers: break
            t_cid = random.choice(trainers)
            t_name = cards[t_cid]['Card Name']
            if can_add(t_name):
                add_card(t_cid, random.randint(1, min(4 - name_counts.get(t_name, 0), target_size - len(deck))))

        # 3. Add Energies (fill remainder to exactly 60)
        types_in_deck = [cards[cid]['Type'] for cid in deck if cards[cid]['Type'] not in ('n/a', '', '{C}')]
        main_type = max(set(types_in_deck), key=types_in_deck.count) if types_in_deck else None
        
        matching_energies = [e for e in energies if cards[e]['Type'] == main_type]
        if not matching_energies and energies:
            matching_energies = energies
            
        if matching_energies:
            e_cid = random.choice(matching_energies)
            while len(deck) < 60:
                add_card(e_cid, 1)

        # Write deck to file
        deck_path = os.path.join(output_dir, f'gen_deck_{i:03d}.csv')
        with open(deck_path, 'w') as f:
            for cid in deck:
                f.write(f"{cid}\n")
                
    print(f"Generated {num_decks} random valid decks in '{output_dir}'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    else:
        num = 100
        
    csv_path = "/home/akhyar/Dokumen/Code/python/new_tcg/agent_rl/EN_Card_Data.csv"
    out_dir = "/home/akhyar/Dokumen/Code/python/new_tcg/agent_rl/deck_generated"
    generate_random_decks(csv_path, out_dir, num)
