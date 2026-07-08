import numpy as np
import collections
import cg.api as api

MAX_ACTIONS = 200
MAX_ZONE_CARDS = 60 

# ==========================================
# PRE-COMPUTE: DATABASE KARTU & ENERGI
# ==========================================
max_energy_req = collections.defaultdict(lambda: collections.defaultdict(float))
card_db = {}
try:
    cards = api.all_card_data()
    for c in cards:
        card_db[c.cardId] = c
        
    attacks = {atk.attackId: atk for atk in api.all_attack()}
    for card in cards:
        for attack_id in card.attacks:
            if attack_id in attacks:
                atk = attacks[attack_id]
                counts = collections.Counter(atk.energies)
                for e_type, count in counts.items():
                    if count > max_energy_req[card.cardId][e_type]:
                        max_energy_req[card.cardId][e_type] = float(count)
except Exception:
    pass

def extract_features(obs_dict):
    """
    Ekstraksi Komprehensif dengan SOTA Board Features (31 Dimensi):
    [0-2] : Categorical (card_id, tool_id, pre_evolution_id)
    [3-6] : Base Stats (is_present, hp_fraction, damage_counters, appear_this_turn)
    [7-11]: Status Conditions
    [12-23]: Normalized Energies
    [24-28]: Action Readiness
    [29-30]: Type Matchup Flags (Hanya untuk Active)
    """
    global_features = np.zeros(10, dtype=np.float32)
    board_features = np.zeros((2, 6, 31), dtype=np.float32)
    
    card_features = {
        "my_hand": np.zeros(MAX_ZONE_CARDS, dtype=np.int32),
        "my_discard": np.zeros(MAX_ZONE_CARDS, dtype=np.int32),
        "opp_discard": np.zeros(MAX_ZONE_CARDS, dtype=np.int32),
        "my_active_id": 0,
        "opp_active_id": 0
    }
    
    action_mask = np.zeros(MAX_ACTIONS, dtype=bool)

    state = obs_dict.get('current')
    if not state:
        return {"global": global_features, "board": board_features, "cards": card_features}, action_mask

    my_index = state.get('yourIndex', 0)
    opp_index = 1 - my_index
    players = state.get('players', [])
    
    if len(players) != 2:
        return {"global": global_features, "board": board_features, "cards": card_features}, action_mask
        
    me = players[my_index]
    opp = players[opp_index]

    # 1. KARTU (SEQUENCE)
    for i, card in enumerate(me.get('hand', [])):
        if i < MAX_ZONE_CARDS: card_features["my_hand"][i] = card.get('id', 0)
    for i, card in enumerate(me.get('discard', [])):
        if i < MAX_ZONE_CARDS: card_features["my_discard"][i] = card.get('id', 0)
    for i, card in enumerate(opp.get('discard', [])):
        if i < MAX_ZONE_CARDS: card_features["opp_discard"][i] = card.get('id', 0)

    active_me_list = me.get('active', [])
    active_opp_list = opp.get('active', [])
    
    my_active_card = active_me_list[0] if active_me_list else None
    opp_active_card = active_opp_list[0] if active_opp_list else None

    if my_active_card: card_features["my_active_id"] = my_active_card.get('id', 0)
    if opp_active_card: card_features["opp_active_id"] = opp_active_card.get('id', 0)

    # 2. GLOBAL FEATURES
    global_features[0] = state.get('turn', 0) / 100.0
    global_features[1] = state.get('turnActionCount', 0) / 50.0
    global_features[2] = 1.0 if state.get('firstPlayer') == my_index else 0.0
    global_features[3] = 1.0 if state.get('supporterPlayed') else 0.0
    global_features[4] = 1.0 if state.get('energyAttached') else 0.0
    stadium_cards = state.get('stadium', [])
    global_features[5] = stadium_cards[0]['id'] if stadium_cards else 0.0
    global_features[6] = me.get('deckCount', 0) / 60.0
    global_features[7] = opp.get('deckCount', 0) / 60.0
    global_features[8] = len(me.get('prize', [])) / 6.0
    global_features[9] = len(opp.get('prize', [])) / 6.0

    # 3. BOARD FEATURES
    def process_pokemon(pokemon, p_idx, slot_idx, player_state):
        if not pokemon: return
        card_id = pokemon.get('id', 0)
        
        # [0-2] Categorical Features
        board_features[p_idx, slot_idx, 0] = card_id
        
        tools = pokemon.get('tools', [])
        board_features[p_idx, slot_idx, 1] = tools[0]['id'] if tools else 0.0
        
        pre_evos = pokemon.get('preEvolution', [])
        board_features[p_idx, slot_idx, 2] = pre_evos[-1]['id'] if pre_evos else 0.0
        
        # [3-6] Base Stats & Damage
        board_features[p_idx, slot_idx, 3] = 1.0 # is_present
        
        hp = pokemon.get('hp', 0)
        max_hp = pokemon.get('maxHp', 1)
        board_features[p_idx, slot_idx, 4] = hp / (max_hp if max_hp > 0 else 1)
        
        # Absolute Damage Counters (1 counter = 10 damage)
        damage_taken = max_hp - hp
        board_features[p_idx, slot_idx, 5] = max(0, damage_taken) / 10.0
        
        board_features[p_idx, slot_idx, 6] = 1.0 if pokemon.get('appearThisTurn') else 0.0
        
        # [7-11] Special Conditions (Only applies to Active / slot 0)
        if slot_idx == 0:
            board_features[p_idx, slot_idx, 7] = 1.0 if player_state.get('poisoned') else 0.0
            board_features[p_idx, slot_idx, 8] = 1.0 if player_state.get('burned') else 0.0
            board_features[p_idx, slot_idx, 9] = 1.0 if player_state.get('asleep') else 0.0
            board_features[p_idx, slot_idx, 10] = 1.0 if player_state.get('paralyzed') else 0.0
            board_features[p_idx, slot_idx, 11] = 1.0 if player_state.get('confused') else 0.0

        # [12-23] Normalized Energies
        attached_counts = collections.Counter(pokemon.get('energies', []))
        req_dict = max_energy_req[card_id]
        for e_type in range(12):
            jumlah_nempel = attached_counts.get(e_type, 0.0)
            dibutuhkan = req_dict.get(e_type, 0.0)
            if dibutuhkan > 0:
                board_features[p_idx, slot_idx, 12 + e_type] = jumlah_nempel / dibutuhkan
            else:
                board_features[p_idx, slot_idx, 12 + e_type] = jumlah_nempel / 5.0

    # Isi Board Kita
    if my_active_card: process_pokemon(my_active_card, 0, 0, me)
    for i, p in enumerate(me.get('bench', [])):
        if i + 1 < 6: process_pokemon(p, 0, i + 1, me)

    # Isi Board Lawan
    if opp_active_card: process_pokemon(opp_active_card, 1, 0, opp)
    for i, p in enumerate(opp.get('bench', [])):
        if i + 1 < 6: process_pokemon(p, 1, i + 1, opp)

    # [29-30] TYPE MATCHUP (Khusus Active vs Active)
    if my_active_card and opp_active_card and card_db:
        my_db_card = card_db.get(my_active_card.get('id', 0))
        opp_db_card = card_db.get(opp_active_card.get('id', 0))
        
        if my_db_card and opp_db_card:
            # Hitungan Kita Hajar Lawan
            if opp_db_card.weakness == my_db_card.energyType:
                board_features[0, 0, 29] = 1.0 # Kita hitting weakness lawan
            if opp_db_card.resistance == my_db_card.energyType:
                board_features[0, 0, 30] = 1.0 # Kita hitting resistance lawan
                
            # Hitungan Lawan Hajar Kita
            if my_db_card.weakness == opp_db_card.energyType:
                board_features[1, 0, 29] = 1.0 # Lawan hitting weakness kita
            if my_db_card.resistance == opp_db_card.energyType:
                board_features[1, 0, 30] = 1.0 # Lawan hitting resistance kita

    # [24-28] ACTION MASKING & 4 KOLOM AKSI
    select_data = obs_dict.get('select')
    if select_data and select_data.get('option'):
        options = select_data['option']
        num_options = min(len(options), MAX_ACTIONS)
        action_mask[:num_options] = True
        
        for opt in options:
            area = opt.get('area')
            idx = opt.get('index', 0)
            
            slot_idx = -1
            if area == 4: slot_idx = 0
            elif area == 5 and idx < 5: slot_idx = idx + 1
                
            if slot_idx != -1:
                opt_type = opt.get('type')
                if opt_type == 13: # ATTACK
                    if board_features[0, slot_idx, 24] == 0.0: board_features[0, slot_idx, 24] = 1.0
                    else: board_features[0, slot_idx, 25] = 1.0
                elif opt_type == 10: # ABILITY
                    if board_features[0, slot_idx, 26] == 0.0: board_features[0, slot_idx, 26] = 1.0
                    else: board_features[0, slot_idx, 27] = 1.0
                elif opt_type == 12: # RETREAT
                    board_features[0, slot_idx, 28] = 1.0

    return {
        "global": global_features,
        "board": board_features,
        "cards": card_features
    }, action_mask
