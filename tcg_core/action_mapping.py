import numpy as np

# =========================================================
# Konfigurasi Pemetaan Baku 250 Aksi (Fixed Action Space)
# Mengacu pada dokumen output_sistem.md (Pendekatan 1)
# =========================================================

NUM_ACTIONS = 250

# Rentang Aksi Kartu (Sesuai dengan OptionType dari C++)
PLAY_START, PLAY_END = 0, 19                 # OptionType.PLAY (Mainkan kartu dari Hand, max 20)
CARD_BOARD_START, CARD_BOARD_END = 20, 31    # OptionType.CARD di Arena (12 slot)
CARD_HAND_START, CARD_HAND_END = 32, 51      # OptionType.CARD di Hand (20 slot)
CARD_DECK_START, CARD_DECK_END = 52, 111     # OptionType.CARD di Deck (60 slot)
CARD_DISCARD_START, CARD_DISCARD_END = 112, 141 # OptionType.CARD di Discard sendiri (30 slot)
CARD_OPP_DISCARD_START, CARD_OPP_DISCARD_END = 142, 171 # OptionType.CARD di Discard lawan (30 slot)

ATTACH_START, ATTACH_END = 172, 183  # OptionType.ATTACH (12 slot board)
EVOLVE_START, EVOLVE_END = 184, 195  # OptionType.EVOLVE (12 slot board)

# Aksi Game Spesifik
ACTION_END = 196                     # OptionType.END
ACTION_RETREAT = 197                 # OptionType.RETREAT
ATTACK_START, ATTACK_END = 198, 203  # OptionType.ATTACK (6 slot)
ABILITY_START, ABILITY_END = 204, 215 # OptionType.ABILITY (12 slot)
ACTION_YES = 216                     # OptionType.YES
ACTION_NO = 217                      # OptionType.NO

ENERGY_START, ENERGY_END = 218, 227  # OptionType.ENERGY, ENERGY_CARD, TOOL_CARD (10 slot)
SKILL_START, SKILL_END = 228, 237    # OptionType.SKILL (10 slot)
NUMBER_START, NUMBER_END = 238, 243  # OptionType.NUMBER (6 slot)
OTHER_START, OTHER_END = 244, 249    # OptionType.DISCARD, etc (6 slot)

def get_action_index_for_option(option: dict, option_list_index: int = 0) -> int:
    """
    Memetakan satu objek 'option' dari C++ engine ke indeks JAX baku (0-249).
    """
    opt_type = option.get("type", "").upper()
    opt_idx_raw = option.get("index")
    opt_idx = int(opt_idx_raw) if opt_idx_raw is not None else 0

    if opt_type == "PLAY":
        return min(PLAY_START + opt_idx, PLAY_END)
        
    elif opt_type == "CARD":
        area = option.get("area")
        area_str = str(area).upper()
        player = int(option.get("playerIndex", 0))
        
        if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"]:
            slot = 6 if player == 0 else 0
            return min(CARD_BOARD_START + slot, CARD_BOARD_END)
        elif area_str in ["BENCH", "5", "AREATYPE.BENCH"]:
            slot = (7 if player == 0 else 1) + opt_idx
            return min(CARD_BOARD_START + slot, CARD_BOARD_END)
        elif area_str in ["DECK", "1", "AREATYPE.DECK"]:
            return min(CARD_DECK_START + opt_idx, CARD_DECK_END)
        elif area_str in ["HAND", "2", "AREATYPE.HAND"]:
            return min(CARD_HAND_START + opt_idx, CARD_HAND_END)
        elif area_str in ["DISCARD", "3", "AREATYPE.DISCARD"]:
            if player == 0:
                return min(CARD_DISCARD_START + opt_idx, CARD_DISCARD_END)
            else:
                return min(CARD_OPP_DISCARD_START + opt_idx, CARD_OPP_DISCARD_END)
        else:
            # Fallback
            return min(CARD_DECK_START + opt_idx, CARD_DECK_END)
            
    elif opt_type == "ATTACH" or opt_type == "EVOLVE":
        area = option.get("inPlayArea")
        idx = int(option.get("inPlayIndex", 0) if option.get("inPlayIndex") is not None else 0)
        area_str = str(area).upper()
        
        # Format slot yang sama dengan CARD_BOARD
        # (Opp Active = 0, Opp Bench = 1-5, My Active = 6, My Bench = 7-11)
        # Tapi biasanya ATTACH/EVOLVE itu untuk My Board (player 0)
        target_slot = 6 # Default My Active
        if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"]:
            target_slot = 6
        elif area_str in ["BENCH", "5", "AREATYPE.BENCH"]:
            target_slot = 7 + idx
            
        start = ATTACH_START if opt_type == "ATTACH" else EVOLVE_START
        end = ATTACH_END if opt_type == "ATTACH" else EVOLVE_END
        return min(start + target_slot, end)
        
    elif opt_type == "END":
        return ACTION_END
        
    elif opt_type == "RETREAT":
        return ACTION_RETREAT
        
    elif opt_type == "ATTACK":
        return min(ATTACK_START + option_list_index, ATTACK_END)
        
    elif opt_type == "ABILITY":
        area = option.get("area")
        idx = int(option.get("index", 0) if option.get("index") is not None else 0)
        area_str = str(area).upper()
        
        target_slot = 6
        if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"]:
            target_slot = 6
        elif area_str in ["BENCH", "5", "AREATYPE.BENCH"]:
            target_slot = 7 + idx
            
        return min(ABILITY_START + target_slot, ABILITY_END)
        
    elif opt_type == "YES":
        return ACTION_YES
        
    elif opt_type == "NO":
        return ACTION_NO
        
    elif opt_type in ["ENERGY", "ENERGY_CARD", "TOOL_CARD"]:
        energy_idx = int(option.get("energyIndex", 0) if option.get("energyIndex") is not None else 0)
        tool_idx = int(option.get("toolIndex", 0) if option.get("toolIndex") is not None else 0)
        item_idx = energy_idx if opt_type in ["ENERGY", "ENERGY_CARD"] else tool_idx
        return min(ENERGY_START + item_idx, ENERGY_END)
        
    elif opt_type == "DISCARD":
        return min(OTHER_START + opt_idx, OTHER_END)
        
    elif opt_type == "SKILL":
        return min(SKILL_START + opt_idx, SKILL_END)
        
    elif opt_type == "NUMBER":
        return min(NUMBER_START + opt_idx, NUMBER_END)
    
    return min(OTHER_START + opt_idx, OTHER_END)

def create_action_mask(select_data: dict, min_count: int = 1, max_count: int = 1) -> np.ndarray:
    """
    Menerima list Option C++ (select_data) dan mengembalikan 
    Numpy Float32 Array berukuran 250 (1.0 = Legal, 0.0 = Ilegal).
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    options = select_data.get("options", [])
    
    for i, option in enumerate(options):
        idx = get_action_index_for_option(option, i)
        mask[idx] = 1.0
        
    # Izinkan aksi END (berhenti memilih) jika batas minimum pemilihan kurang dari maksimum
    if min_count < max_count:
        mask[ACTION_END] = 1.0
        
    # Perlindungan Failsafe: Jika C++ tidak mereturn opsi legal apapun,
    # kita paksa End Turn menjadi legal agar JAX tidak mengalami error/NaN pada softmax.
    if np.sum(mask) == 0:
        mask[ACTION_END] = 1.0
        
    return mask


def decode_action(sorted_action_indices: list, select_data: dict, min_count: int = 1, max_count: int = 1) -> list:
    """
    Mengonversi daftar aksi terurut yang dipilih AI (berdasarkan probabilitas tertinggi)
    menjadi list pilihan indeks (0-n) sesuai dengan opsi legal dari C++ engine.
    Mendukung multiple-choice (misalnya membuang 2 kartu) dengan mengumpulkan beberapa
    opsi teratas hingga memenuhi `min_count` yang disyaratkan engine.
    """
    options = select_data.get("options", [])
    if not options:
        return []

    # Map setiap opsi C++ ke indeks JAX-nya (0-249)
    cpp_option_to_jax_idx = []
    for i, option in enumerate(options):
        cpp_option_to_jax_idx.append((i, get_action_index_for_option(option, i)))

    choices = []
    # 1. Telusuri pilihan AI dari probabilitas terbesar ke terkecil
    for jax_idx in sorted_action_indices:
        if jax_idx == 160: # ACTION_END
            if len(choices) >= min_count:
                break # Model memilih berhenti dan kuota minimum sudah terpenuhi
            else:
                continue # Belum memenuhi minimum, abaikan END

        # Cari apakah ada opsi C++ yang cocok dengan pilihan JAX ini
        for cpp_idx, mapped_jax_idx in cpp_option_to_jax_idx:
            if mapped_jax_idx == jax_idx and cpp_idx not in choices:
                choices.append(cpp_idx)
                break # Pindah ke JAX idx selanjutnya (satu JAX idx untuk satu opsi C++)
        
        if len(choices) >= max_count:
            break

    # 2. Fallback jika masih belum memenuhi min_count
    if len(choices) < min_count:
        for cpp_idx in range(len(options)):
            if len(choices) >= min_count:
                break
            if cpp_idx not in choices:
                choices.append(cpp_idx)

    return choices
