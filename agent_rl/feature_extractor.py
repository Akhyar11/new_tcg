import numpy as np
from .action_mapping import create_action_mask, NUM_ACTIONS
from cg.api import all_card_data, all_attack, State, SelectData, PlayerState, Pokemon, Card, OptionType

# Memuat data statis secara global untuk mempercepat lookup saat simulasi berjalan
CARD_DB = {c.cardId: c for c in all_card_data()}
ATTACK_DB = {a.attackId: a for a in all_attack()}

def _get_val(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_card(card, is_active=False, player_state: PlayerState = None) -> np.ndarray:
    """
    Mengonversi objek `Card` atau `Pokemon` (bisa dict atau dataclass) menjadi tensor (31,).
    """
    features = np.zeros(31, dtype=np.float32)
    if not card:
        return features

    # 3 ID Dasar
    card_id = _get_val(card, 'id', 0)
    features[0] = card_id
    card_data = CARD_DB.get(card_id)

    # Periksa apakah ini objek Pokemon di Arena atau Card biasa
    # Pokemon memiliki atribut hp atau key 'hp'
    hp_val = _get_val(card, 'hp')
    if hp_val is not None:
        # --- Objek Pokemon ---
        tools = _get_val(card, 'tools', [])
        pre_evo = _get_val(card, 'preEvolution', [])
        features[1] = _get_val(tools[0], 'id', 0) if tools else 0
        features[2] = _get_val(pre_evo[0], 'id', 0) if pre_evo else 0

        energies = _get_val(card, 'energies', [])
        attached_counts = {k: 0 for k in range(12)}
        for e in energies:
            attached_counts[int(e)] += 1

        # 1. Hitung kebutuhan energi (cost) maksimal dari serangan Pokemon
        total_cost = cost_g = cost_r = cost_w = cost_l = cost_other = 0
        if card_data and card_data.attacks:
            for atk_id in card_data.attacks:
                atk = ATTACK_DB.get(atk_id)
                if atk:
                    atk_g = atk_r = atk_w = atk_l = atk_other = 0
                    for e in atk.energies:
                        e_val = int(e)
                        if e_val == 1: atk_g += 1
                        elif e_val == 2: atk_r += 1
                        elif e_val == 3: atk_w += 1
                        elif e_val == 4: atk_l += 1
                        else: atk_other += 1

                    if len(atk.energies) > total_cost:
                        total_cost = len(atk.energies)
                        cost_g, cost_r, cost_w, cost_l, cost_other = atk_g, atk_r, atk_w, atk_l, atk_other

        # 2. Fitur energi terpasang dibagi dengan KEBUTUHAN (progress ratio)
        # Menggunakan max(cost, 1.0) untuk menghindari pembagian dengan 0
        features[3] = len(energies) / max(total_cost, 1.0)
        features[4] = attached_counts[1] / max(cost_g, 1.0) # Grass
        features[5] = attached_counts[2] / max(cost_r, 1.0) # Fire
        features[6] = attached_counts[3] / max(cost_w, 1.0) # Water
        features[7] = attached_counts[4] / max(cost_l, 1.0) # Lightning
        features[8] = sum([attached_counts[k] for k in [5,6,7,8,9,10,11,0]]) / max(cost_other, 1.0)

        features[9] = total_cost / 10.0
        features[10] = cost_g / 10.0
        features[11] = cost_r / 10.0
        features[12] = cost_w / 10.0
        features[13] = cost_l / 10.0
        features[14] = cost_other / 10.0

        features[15] = 1.0 # is_present
        max_hp_val = _get_val(card, 'maxHp', 0)
        if max_hp_val > 0:
            features[16] = hp_val / max_hp_val
            features[17] = (max_hp_val - hp_val) / 300.0
        features[18] = 1.0 if _get_val(card, 'appearThisTurn') else 0.0

        if is_active and player_state:
            features[19] = 1.0 if _get_val(player_state, 'poisoned') else 0.0
            features[20] = 1.0 if _get_val(player_state, 'burned') else 0.0
            features[21] = 1.0 if _get_val(player_state, 'asleep') else 0.0
            features[22] = 1.0 if _get_val(player_state, 'paralyzed') else 0.0
            features[23] = 1.0 if _get_val(player_state, 'confused') else 0.0

    else:
        # --- Objek Card Biasa ---
        features[15] = 1.0 # is_present

    return features


def fill_sequence(sequence, start_idx, max_len, item_list, is_active=False, player_state=None):
    if not item_list:
        return
    for i in range(min(len(item_list), max_len)):
        if item_list[i]:
            sequence[start_idx + i] = parse_card(item_list[i], is_active, player_state)


def extract_features(state: State, select_data: SelectData, your_index: int, opp_known_hand: list = None) -> dict:
    if opp_known_hand is None: opp_known_hand = []
    
    # 1. CARD EMBEDDING SEQUENCE (113, 31)
    seq_input = np.zeros((113, 31), dtype=np.float32)

    my_state = state.players[your_index]
    opp_index = 1 - your_index
    opp_state = state.players[opp_index]

    # Slot 0-19: My Hand (20)
    fill_sequence(seq_input, 0, 20, my_state.hand)
    # Slot 20-49: My Discard (30)
    fill_sequence(seq_input, 20, 30, my_state.discard)
    # Slot 50-79: Opp Discard (30)
    fill_sequence(seq_input, 50, 30, opp_state.discard)

    # Slot 80-85: My Board (Active + Bench)
    fill_sequence(seq_input, 80, 1, my_state.active, is_active=True, player_state=my_state)
    fill_sequence(seq_input, 81, 5, my_state.bench)

    # Slot 86-91: Opp Board (Active + Bench)
    fill_sequence(seq_input, 86, 1, opp_state.active, is_active=True, player_state=opp_state)
    fill_sequence(seq_input, 87, 5, opp_state.bench)

    # Slot 92: Stadium (1)
    if state.stadium:
        seq_input[92] = parse_card(state.stadium[0])

    # Slot 93-112: Opponent Known Hand (20)
    fill_sequence(seq_input, 93, 20, opp_known_hand)

    # 2. GLOBAL STATE (266)
    glob_input = np.zeros(266, dtype=np.float32)
    glob_input[0] = state.turn / 100.0
    glob_input[1] = state.turnActionCount / 50.0
    glob_input[2] = 1.0 if state.firstPlayer == your_index else 0.0
    glob_input[3] = 1.0 if state.supporterPlayed else 0.0
    glob_input[4] = 1.0 if state.energyAttached else 0.0
    glob_input[5] = 1.0 if state.retreated else 0.0

    my_board_count = (1 if my_state.active and my_state.active[0] else 0) + len(my_state.bench)
    opp_board_count = (1 if opp_state.active and opp_state.active[0] else 0) + len(opp_state.bench)
    glob_input[6] = my_board_count / 6.0
    glob_input[7] = opp_board_count / 6.0

    glob_input[8] = my_state.deckCount / 60.0
    glob_input[9] = opp_state.deckCount / 60.0
    glob_input[10] = len(my_state.prize) / 6.0
    glob_input[11] = len(opp_state.prize) / 6.0

    # minCount / maxCount dari SelectData (biar model tahu perlu pilih 1 atau N)
    if select_data is not None:
        glob_input[12] = select_data.minCount / 10.0
        glob_input[13] = select_data.maxCount / 10.0
    else:
        glob_input[12] = 0.0
        glob_input[13] = 0.0

    # JAX Action Mask Processing
    # Convert SelectData objects into python dict to match create_action_mask prototype
    try:
        if select_data is not None and select_data.option is not None:
            mock_select_dict = {"options": [{"type": OptionType(o.type).name, "index": o.index} for o in select_data.option]}
        else:
            mock_select_dict = {"options": []}
    except ValueError:
        # Fallback if unknown option type
        mock_select_dict = {"options": []}

    glob_input[16:16+NUM_ACTIONS] = create_action_mask(mock_select_dict)

    return {
        "seq_input": seq_input,
        "glob_input": glob_input
    }
