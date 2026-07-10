import numpy as np

# =========================================================
# Konfigurasi Pemetaan Baku 250 Aksi (Fixed Action Space)
# Mengacu pada dokumen output_sistem.md (Pendekatan 1)
# =========================================================

NUM_ACTIONS = 250

# Rentang Aksi Kartu (Sesuai dengan OptionType dari C++)
PLAY_START, PLAY_END = 0, 59         # OptionType.PLAY (Mainkan kartu dari Hand)
CARD_START, CARD_END = 60, 119       # OptionType.CARD (Pilih kartu spesifik)
ATTACH_START, ATTACH_END = 120, 139  # OptionType.ATTACH
EVOLVE_START, EVOLVE_END = 140, 159  # OptionType.EVOLVE

# Aksi Game Spesifik
ACTION_END = 160                     # OptionType.END
ACTION_RETREAT = 161                 # OptionType.RETREAT
ATTACK_START, ATTACK_END = 162, 167  # OptionType.ATTACK
ABILITY_START, ABILITY_END = 168, 179 # OptionType.ABILITY
ACTION_YES = 180                     # OptionType.YES
ACTION_NO = 181                      # OptionType.NO
ENERGY_START, ENERGY_END = 182, 199  # OptionType.ENERGY, ENERGY_CARD
SKILL_START, SKILL_END = 200, 219    # OptionType.SKILL
NUMBER_START, NUMBER_END = 220, 239  # OptionType.NUMBER

# Sisa 240-249 tersedia untuk tipe lain (misal: TOOL_CARD, DISCARD, SPECIAL_CONDITION)
OTHER_START, OTHER_END = 240, 249

def get_action_index_for_option(option: dict) -> int:
    """
    Memetakan satu objek 'option' dari C++ engine ke indeks JAX baku (0-249).
    """
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
    return min(OTHER_START + opt_idx, OTHER_END)


def create_action_mask(select_data: dict) -> np.ndarray:
    """
    Menerima list Option C++ (select_data) dan mengembalikan 
    Numpy Float32 Array berukuran 250 (1.0 = Legal, 0.0 = Ilegal).
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    options = select_data.get("options", [])
    
    for option in options:
        idx = get_action_index_for_option(option)
        mask[idx] = 1.0
        
    # Perlindungan Failsafe: Jika C++ tidak mereturn opsi legal apapun (hal ini aneh),
    # kita paksa End Turn menjadi legal agar JAX tidak mengalami error/NaN pada softmax.
    if np.sum(mask) == 0:
        mask[ACTION_END] = 1.0
        
    return mask


def decode_action(sorted_action_indices: list, select_data: dict, min_count: int = 1) -> list:
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
        cpp_option_to_jax_idx.append((i, get_action_index_for_option(option)))

    choices = []
    # 1. Telusuri pilihan AI dari probabilitas terbesar ke terkecil
    for jax_idx in sorted_action_indices:
        # Cari apakah ada opsi C++ yang cocok dengan pilihan JAX ini
        for cpp_idx, mapped_jax_idx in cpp_option_to_jax_idx:
            if mapped_jax_idx == jax_idx and cpp_idx not in choices:
                choices.append(cpp_idx)
                break # Pindah ke JAX idx selanjutnya (satu JAX idx untuk satu opsi C++)
        
        if len(choices) >= min_count:
            break

    # 2. Fallback jika masih belum memenuhi min_count (karena model AI masih bodoh/belum konvergen)
    if len(choices) < min_count:
        for cpp_idx in range(len(options)):
            if cpp_idx not in choices:
                choices.append(cpp_idx)
            if len(choices) >= min_count:
                break

    return choices
