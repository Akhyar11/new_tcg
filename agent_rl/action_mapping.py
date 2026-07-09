import numpy as np

# =========================================================
# Konfigurasi Pemetaan Baku 250 Aksi (Fixed Action Space)
# Mengacu pada dokumen output_sistem.md (Pendekatan 1)
# =========================================================

NUM_ACTIONS = 250

# Rentang Aksi Kartu
HAND_START, HAND_END = 0, 59
DECK_START, DECK_END = 60, 119
DISCARD_START, DISCARD_END = 120, 179

# Aksi Game Spesifik
ACTION_END_TURN = 180
ACTION_RETREAT = 181
ATTACK_START, ATTACK_END = 182, 185
ABILITY_START, ABILITY_END = 186, 197
TARGET_MY_BOARD_START, TARGET_MY_BOARD_END = 198, 209
TARGET_OPP_BOARD_START, TARGET_OPP_BOARD_END = 210, 221
ACTION_YES = 222
ACTION_NO = 223
ENERGY_START, ENERGY_END = 224, 230

# Sisa 231-249 tersedia untuk pengembangan (misal: Pilih Target Tipe Lain, Select Stadium)


def get_action_index_for_option(option: dict) -> int:
    """
    Memetakan satu objek 'option' dari C++ engine ke indeks JAX baku (0-249).
    Kita mengasumsikan 'option' adalah dictionary (hasil parsing JSON) yang
    memiliki kunci 'type' (string) dan 'index' (int).
    """
    opt_type = option.get("type", "").upper()
    opt_idx = int(option.get("index", 0))

    if opt_type == "HAND":
        return min(HAND_START + opt_idx, HAND_END)
    elif opt_type == "DECK":
        return min(DECK_START + opt_idx, DECK_END)
    elif opt_type == "DISCARD":
        return min(DISCARD_START + opt_idx, DISCARD_END)
    elif opt_type == "END_TURN":
        return ACTION_END_TURN
    elif opt_type == "RETREAT":
        return ACTION_RETREAT
    elif opt_type == "ATTACK":
        return min(ATTACK_START + opt_idx, ATTACK_END)
    elif opt_type == "ABILITY":
        return min(ABILITY_START + opt_idx, ABILITY_END)
    elif opt_type in ["TARGET_MY_BOARD", "MY_BOARD"]:
        return min(TARGET_MY_BOARD_START + opt_idx, TARGET_MY_BOARD_END)
    elif opt_type in ["TARGET_OPP_BOARD", "OPP_BOARD"]:
        return min(TARGET_OPP_BOARD_START + opt_idx, TARGET_OPP_BOARD_END)
    elif opt_type == "YES":
        return ACTION_YES
    elif opt_type == "NO":
        return ACTION_NO
    elif opt_type == "ENERGY":
        return min(ENERGY_START + opt_idx, ENERGY_END)
    
    # Fallback untuk tipe tak dikenal (menggunakan indeks aman terakhir)
    return 249


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
        mask[ACTION_END_TURN] = 1.0
        
    return mask


def decode_action(action_index: int, select_data: dict, max_count: int = 1) -> list:
    """
    Mengonversi kembali indeks JAX (0-249) yang dipilih AI menjadi format 
    list `[int]` yang menunjuk ke indeks aktual di array 'options' milik C++ engine.
    
    Contoh: Jika model JAX memilih aksi 180 (End Turn), fungsi ini mencari
    'option' mana di dalam select_data yang merupakan End Turn, lalu mereturn
    posisi indeks 'option' tersebut di dalam list C++ (untuk diteruskan ke search_step).
    """
    options = select_data.get("options", [])
    
    # Mencari semua opsi C++ yang memetakan ke action_index ini
    matched_option_indices = []
    for i, option in enumerate(options):
        if get_action_index_for_option(option) == action_index:
            matched_option_indices.append(i)
            
    # Mengembalikan daftar indeks C++ yang valid (dibatasi oleh max_count untuk Multi-Aksi)
    if matched_option_indices:
        return matched_option_indices[:max_count]
    
    # Fallback/Safe-Return jika AI memilih aksi ilegal (hal yang harusnya 
    # dicegah oleh action_mask, tapi tetap perlu di-handle demi stabilitas).
    return [0] if options else []
