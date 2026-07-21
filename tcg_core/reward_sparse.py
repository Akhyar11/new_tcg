"""
Reward System — Sparse (Win/Loss only) untuk PPO Self-Play.

Digunakan saat agen AI sudah pintar dan tidak perlu dituntun
dengan intermediate rewards (seperti evolve, attach energy, dll).
Agen dipaksa mencari cara ter-efisien untuk menang murni dari
Win/Loss signals.
"""
import numpy as np

def reset_trackers():
    """Reset event counters untuk game baru."""
    pass

def detect_events(old_state, new_state, player_index: int, logs: list = None) -> dict:
    """
    Mendeteksi event apa yang terjadi dalam satu step.
    Pada mode sparse, ini sengaja dikosongkan untuk menghemat komputasi
    karena kita tidak menggunakan intermediate events untuk reward.
    """
    return {}

def calculate_step_reward(new_state, player_index: int, events: dict = None, end_reason: int = 0, turn_changed: bool = False) -> float:
    """
    Reward dengan skala Sparse:
    Menang = +1.0
    Kalah = -1.0
    Draw/Timeout = -0.5 (Penalti kecil agar agen tidak membuang-buang waktu)
    Step biasa = 0.0
    """
    if new_state is None:
        return 0.0

    if new_state.result != -1:
        won = (new_state.result == player_index)
        lost = (new_state.result == (1 - player_index))
        draw = (new_state.result == 2)

        if won:
            return 1.0
        elif lost:
            return -1.0
        elif draw:
            return -0.5 # Mencegah agen dengan sengaja bermain aman sampai timeout

    return 0.0
