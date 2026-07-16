with open("agent_rl/action_mapping.py", "r") as f:
    content = f.read()

import re

old_func = """def get_action_index_for_option(option: dict) -> int:
    \"\"\"
    Memetakan satu objek 'option' dari C++ engine ke indeks JAX baku (0-249).
    \"\"\"
    opt_type = option.get("type", "").upper()
    opt_idx_raw = option.get("index")
    opt_idx = int(opt_idx_raw) if opt_idx_raw is not None else 0

    if opt_type == "PLAY":
        return min(PLAY_START + opt_idx, PLAY_END)
    elif opt_type == "CARD":
        return min(CARD_START + opt_idx, CARD_END)
    elif opt_type == "ATTACH":
        return min(ATTACH_START + opt_idx, ATTACH_END)
    elif opt_type == "EVOLVE":
        return min(EVOLVE_START + opt_idx, EVOLVE_END)
    elif opt_type == "END":
        return ACTION_END
    elif opt_type == "RETREAT":
        return ACTION_RETREAT
    elif opt_type == "ATTACK":
        return min(ATTACK_START + opt_idx, ATTACK_END)
    elif opt_type == "ABILITY":
        return min(ABILITY_START + opt_idx, ABILITY_END)
    elif opt_type == "YES":
        return ACTION_YES
    elif opt_type == "NO":
        return ACTION_NO
    elif opt_type in ["ENERGY", "ENERGY_CARD"]:
        return min(ENERGY_START + opt_idx, ENERGY_END)
    elif opt_type == "SKILL":
        return min(SKILL_START + opt_idx, SKILL_END)
    elif opt_type == "NUMBER":
        return min(NUMBER_START + opt_idx, NUMBER_END)
    
    # Fallback untuk tipe tak dikenal (menggunakan indeks aman)
    return min(OTHER_START + opt_idx, OTHER_END)"""

new_func = """def get_action_index_for_option(option: dict, option_list_index: int = 0) -> int:
    \"\"\"
    Memetakan satu objek 'option' dari C++ engine ke indeks JAX baku (0-249).
    \"\"\"
    opt_type = option.get("type", "").upper()
    opt_idx_raw = option.get("index")
    opt_idx = int(opt_idx_raw) if opt_idx_raw is not None else 0

    if opt_type == "PLAY":
        return min(PLAY_START + opt_idx, PLAY_END)
    elif opt_type == "CARD":
        return min(CARD_START + opt_idx, CARD_END)
    elif opt_type == "ATTACH" or opt_type == "EVOLVE":
        area = option.get("inPlayArea")
        idx = int(option.get("inPlayIndex", 0) if option.get("inPlayIndex") is not None else 0)
        area_str = str(area).upper()
        target_slot = 0
        if area_str == "ACTIVE" or area == 4 or area_str == "AREATYPE.ACTIVE":
            target_slot = 0
        elif area_str == "BENCH" or area == 5 or area_str == "AREATYPE.BENCH":
            target_slot = 1 + idx
        start = ATTACH_START if opt_type == "ATTACH" else EVOLVE_START
        end = ATTACH_END if opt_type == "ATTACH" else EVOLVE_END
        return min(start + target_slot, end)
    elif opt_type == "END":
        return ACTION_END
    elif opt_type == "RETREAT":
        return ACTION_RETREAT
    elif opt_type == "ATTACK":
        # Gunakan urutan di dalam list option (0 = Serangan pertama, 1 = Serangan kedua)
        return min(ATTACK_START + option_list_index, ATTACK_END)
    elif opt_type == "ABILITY":
        area = option.get("area")
        idx = int(option.get("index", 0) if option.get("index") is not None else 0)
        area_str = str(area).upper()
        target_slot = 0
        if area_str == "ACTIVE" or area == 4 or area_str == "AREATYPE.ACTIVE":
            target_slot = 0
        elif area_str == "BENCH" or area == 5 or area_str == "AREATYPE.BENCH":
            target_slot = 1 + idx
        return min(ABILITY_START + target_slot, ABILITY_END)
    elif opt_type == "YES":
        return ACTION_YES
    elif opt_type == "NO":
        return ACTION_NO
    elif opt_type in ["ENERGY", "ENERGY_CARD"]:
        return min(ENERGY_START + opt_idx, ENERGY_END)
    elif opt_type == "SKILL":
        return min(SKILL_START + opt_idx, SKILL_END)
    elif opt_type == "NUMBER":
        return min(NUMBER_START + opt_idx, NUMBER_END)
    
    return min(OTHER_START + opt_idx, OTHER_END)"""

content = content.replace(old_func, new_func)

# Fix loop in create_action_mask
old_mask = """    for option in options:
        idx = get_action_index_for_option(option)"""
new_mask = """    for i, option in enumerate(options):
        idx = get_action_index_for_option(option, i)"""
content = content.replace(old_mask, new_mask)

# Fix loop in decode_action
old_decode = """    for i, option in enumerate(options):
        cpp_option_to_jax_idx.append((i, get_action_index_for_option(option)))"""
new_decode = """    for i, option in enumerate(options):
        cpp_option_to_jax_idx.append((i, get_action_index_for_option(option, i)))"""
content = content.replace(old_decode, new_decode)

with open("agent_rl/action_mapping.py", "w") as f:
    f.write(content)

