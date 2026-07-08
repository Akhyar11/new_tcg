import numpy as np
from agent_rl.feature_extractor import extract_features

# 1. BUAT MOCK DATA (Mirip dengan JSON yang dikembalikan engine api.py)
mock_obs_dict = {
    'current': {
        'turn': 5,
        'turnActionCount': 2,
        'yourIndex': 0,
        'firstPlayer': 1,
        'supporterPlayed': True,
        'energyAttached': False,
        'stadium': [{'id': 50}],
        'players': [
            # PLAYER 0 (KITA)
            {
                'deckCount': 40,
                'prize': [None] * 4, # Sisa 4 prize
                'poisoned': False,
                'burned': True,
                'asleep': False,
                'paralyzed': False,
                'confused': False,
                'hand': [{'id': 10}, {'id': 45}, {'id': 12}], # 3 Kartu di tangan
                'discard': [{'id': 5}, {'id': 8}], # 2 Kartu di discard
                'active': [{
                    'id': 4, # Charizard
                    'hp': 120, 'maxHp': 150,
                    'appearThisTurn': False,
                    'energies': [2, 2, 2, 0, 0], # 3 Fire (tipe 2), 2 Colorless (tipe 0)
                    'tools': [{'id': 99}] # Bawa 1 Tool
                }],
                'bench': [{
                    'id': 1, # Bulbasaur
                    'hp': 60, 'maxHp': 60,
                    'appearThisTurn': True,
                    'energies': [1], # 1 Grass (tipe 1)
                    'tools': []
                }]
            },
            # PLAYER 1 (MUSUH)
            {
                'deckCount': 45,
                'prize': [None] * 6, # Sisa 6 prize
                'discard': [{'id': 100}],
                'active': [{
                    'id': 7, # Squirtle
                    'hp': 50, 'maxHp': 50,
                    'appearThisTurn': False,
                    'energies': [3], # 1 Water (tipe 3)
                    'tools': []
                }],
                'bench': []
            }
        ]
    },
    'select': {
        'option': [
            {'type': 13, 'area': 4, 'index': 0, 'attackId': 100}, # Attack 1 Sah (Active Kita)
            {'type': 10, 'area': 5, 'index': 0}, # Ability 1 Sah (Bench Kita indeks 0)
        ]
    }
}

# 2. JALANKAN FUNGSI EKSTRAKSI
features, mask = extract_features(mock_obs_dict)

# 3. CETAK HASILNYa
print("=== 1. HASIL EKSTRAKSI KARTU (SEQUENCE ARRAY) ===")
cards = features['cards']
print("My Hand (Shape {}):".format(cards['my_hand'].shape), cards['my_hand'][:10], "... (padding 0)")
print("My Discard:", cards['my_discard'][:5], "... (padding 0)")
print("Opp Discard:", cards['opp_discard'][:5], "... (padding 0)")
print("My Active Card ID:", cards['my_active_id'])
print("Opp Active Card ID:", cards['opp_active_id'])
print()

print("=== 2. HASIL EKSTRAKSI GLOBAL ===")
glob = features['global']
print("Turn:", glob[0], "| ActionCount:", glob[1], "| MyDeckFraction:", round(glob[6], 2), "| OppDeckFraction:", round(glob[7], 2))
print()

print("=== 3. HASIL EKSTRAKSI BOARD (ACTIVE KITA) ===")
board = features['board']
active_me = board[0, 0] # Player 0, Slot 0
print("Card ID        :", int(active_me[0]))
print("Tool ID        :", int(active_me[1]))
print("HP Fraction    :", active_me[3])
print("Is Burned?     :", active_me[6])
print("Fire Energy Normalized :", round(active_me[10 + 2], 2))
print("Colorless Energy Norm. :", round(active_me[10 + 0], 2))
print("- 4 Kolom Aksi -")
print("Attack 1 Ready :", active_me[22])
print("Attack 2 Ready :", active_me[23])

print("\n=== 4. HASIL EKSTRAKSI BOARD (BENCH 1 KITA) ===")
bench_me = board[0, 1] # Player 0, Slot 1
print("Card ID        :", int(bench_me[0]))
print("Grass Energy Normalized:", round(bench_me[10 + 1], 2))
print("Ability 1 Ready:", bench_me[24])

print("\n=== 5. HASIL ACTION MASK ===")
print("Mask valid pertama:", mask[:3])
