import collections
import cg.api as api

# 1. Bangun Kamus Kebutuhan Maksimal Energi per Kartu
max_energy_req = collections.defaultdict(lambda: collections.defaultdict(float))

cards = api.all_card_data()
attacks = {atk.attackId: atk for atk in api.all_attack()}

# Pre-compute kamus agar cepat saat ekstraksi
for card in cards:
    for attack_id in card.attacks:
        if attack_id in attacks:
            atk = attacks[attack_id]
            counts = collections.Counter(atk.energies)
            for e_type, count in counts.items():
                if count > max_energy_req[card.cardId][e_type]:
                    max_energy_req[card.cardId][e_type] = float(count)

def extract_single_pokemon(pokemon_dict, options_list):
    """
    Indeks 0: card_id
    Indeks 1: tool_id
    Indeks 2: is_present
    Indeks 3: hp_fraction
    Indeks 4: appear_this_turn
    Indeks 5-9: status kondisi
    Indeks 10-21: Energi Ternormalisasi (12 elemen)
    Indeks 22: attack_1_ready
    Indeks 23: attack_2_ready
    Indeks 24: ability_1_ready
    Indeks 25: ability_2_ready
    Indeks 26: can_retreat
    """
    features = [0.0] * 27
    
    card_id = pokemon_dict.get('id', 0)
    features[0] = card_id
    features[2] = 1.0 # is_present
    
    hp = pokemon_dict.get('hp', 0)
    max_hp = pokemon_dict.get('maxHp', 1)
    features[3] = hp / (max_hp if max_hp > 0 else 1)
    
    # NORMALISASI ENERGI
    attached_energies = pokemon_dict.get('energies', [])
    attached_counts = collections.Counter(attached_energies)
    req_dict = max_energy_req[card_id]
    
    for e_type in range(12):
        jumlah_nempel = attached_counts.get(e_type, 0.0)
        dibutuhkan = req_dict.get(e_type, 0.0)
        if dibutuhkan > 0:
            features[10 + e_type] = jumlah_nempel / dibutuhkan
        else:
            features[10 + e_type] = jumlah_nempel / 5.0 # fallback

    # 4 KOLOM AKSI (Membaca dari options_list)
    # Anggap card_id ini memiliki:
    # attackId_1 = 100, attackId_2 = 101
    mock_attack_1_id = 100
    mock_attack_2_id = 101
    
    for opt in options_list:
        opt_type = opt.get('type')
        if opt_type == 13: # ATTACK
            atk_id = opt.get('attackId')
            if atk_id == mock_attack_1_id:
                features[22] = 1.0 # attack_1_ready
            elif atk_id == mock_attack_2_id:
                features[23] = 1.0 # attack_2_ready
        elif opt_type == 10: # ABILITY (SKILL)
            # Biasanya direpresentasikan dengan indeks kemampuannya (0 atau 1)
            skill_idx = opt.get('index', 0)
            if skill_idx == 0:
                features[24] = 1.0 # ability_1_ready
            elif skill_idx == 1:
                features[25] = 1.0 # ability_2_ready
        elif opt_type == 12: # RETREAT
            features[26] = 1.0 # can_retreat
            
    return features

# MOCK DATA
mock_pokemon = {
    'id': 4, # Asumsi Charizard
    'hp': 120, 'maxHp': 150,
    'energies': [2, 2, 2, 0, 0] # 3 Fire (2), 2 Colorless (0)
}

# Misal game mensimulasikan bahwa energi sudah cukup untuk Attack 1, dan Ability 1 bisa dipakai
mock_options = [
    {'type': 13, 'attackId': 100}, # Attack 1 Sah
    {'type': 10, 'index': 0},      # Ability 1 Sah
    {'type': 12}                   # Retreat Sah
]

hasil = extract_single_pokemon(mock_pokemon, mock_options)

print("Card ID:", hasil[0])
print("HP Fraction:", hasil[3])
print("Fire Energy Normalized:", hasil[10 + 2])
print("Colorless Energy Normalized:", hasil[10 + 0])
print("--- 4 KOLOM AKSI ---")
print("Attack 1 Ready :", hasil[22])
print("Attack 2 Ready :", hasil[23])
print("Ability 1 Ready:", hasil[24])
print("Ability 2 Ready:", hasil[25])
print("Can Retreat    :", hasil[26])
