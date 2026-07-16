with open("agent_rl/action_mapping.py", "r") as f:
    lines = f.readlines()

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
        # Untuk memilih CARD (misal dari deck, hand, atau di arena)
        # Jika memilih di arena (ACTIVE/BENCH), bedakan berdasarkan slot target
        area = option.get("area")
        area_str = str(area).upper()
        if area_str in ["ACTIVE", "BENCH", "4", "5", "AREATYPE.ACTIVE", "AREATYPE.BENCH"]:
            player = int(option.get("playerIndex", 0))
            # Slot: 0 = Opponent Active, 1-5 = Opponent Bench, 6 = My Active, 7-11 = My Bench
            slot = (6 if player == 0 else 0) + (0 if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"] else (1 + opt_idx))
            return min(CARD_START + slot, CARD_END)
        else:
            # Dari Deck / Hand / Discard (maksimal 60)
            return min(CARD_START + 12 + opt_idx, CARD_END)
            
    elif opt_type == "ATTACH" or opt_type == "EVOLVE":
        # Target area dan target index
        area = option.get("inPlayArea")
        idx = int(option.get("inPlayIndex", 0) if option.get("inPlayIndex") is not None else 0)
        area_str = str(area).upper()
        target_slot = 0
        if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"]:
            target_slot = 0
        elif area_str in ["BENCH", "5", "AREATYPE.BENCH"]:
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
        if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"]:
            target_slot = 0
        elif area_str in ["BENCH", "5", "AREATYPE.BENCH"]:
            target_slot = 1 + idx
        return min(ABILITY_START + target_slot, ABILITY_END)
        
    elif opt_type == "YES":
        return ACTION_YES
        
    elif opt_type == "NO":
        return ACTION_NO
        
    elif opt_type in ["ENERGY", "ENERGY_CARD", "TOOL_CARD"]:
        # Biasanya memilih energi/tool spesifik yang terpasang di pokemon
        # Karena bisa ada banyak, kita pakai energyIndex / toolIndex
        energy_idx = int(option.get("energyIndex", 0) if option.get("energyIndex") is not None else 0)
        tool_idx = int(option.get("toolIndex", 0) if option.get("toolIndex") is not None else 0)
        item_idx = energy_idx if opt_type in ["ENERGY", "ENERGY_CARD"] else tool_idx
        return min(ENERGY_START + item_idx, ENERGY_END)
        
    elif opt_type == "DISCARD":
        # Discard kartu di arena (misal stadium)
        return min(OTHER_START + opt_idx, OTHER_END)
        
    elif opt_type == "SKILL":
        return min(SKILL_START + opt_idx, SKILL_END)
        
    elif opt_type == "NUMBER":
        return min(NUMBER_START + opt_idx, NUMBER_END)
    
    return min(OTHER_START + opt_idx, OTHER_END)
"""

# Find start and end of get_action_index_for_option
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("def get_action_index_for_option"):
        start_idx = i
    elif start_idx != -1 and line.startswith("def create_action_mask"):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    lines = lines[:start_idx] + [new_func + "\n"] + lines[end_idx:]
    with open("agent_rl/action_mapping.py", "w") as f:
        f.writelines(lines)
    print("Action mapping replaced successfully.")
else:
    print("Function not found.")
