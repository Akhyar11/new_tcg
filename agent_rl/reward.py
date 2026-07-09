import numpy as np

def calc_potential(state, your_index: int) -> float:
    """
    Menghitung Potential Function (Phi) untuk reward shaping.
    Ini membantu mencegah reward hacking (seperti damage/heal looping).
    """
    if state is None:
        return 0.0
        
    opp_index = 1 - your_index
    my_state = state.players[your_index]
    opp_state = state.players[opp_index]
    
    # 1. Prize Cards (Selisih)
    my_prize_taken = 6 - len(my_state.prize)
    opp_prize_taken = 6 - len(opp_state.prize)
    prize_diff = my_prize_taken - opp_prize_taken
    
    # 2. Selisih Total HP di papan
    my_hp = sum([p.hp for p in my_state.active if p]) + sum([p.hp for p in my_state.bench])
    opp_hp = sum([p.hp for p in opp_state.active if p]) + sum([p.hp for p in opp_state.bench])
    hp_diff = my_hp - opp_hp
    
    # 3. Selisih sisa kartu di Deck (menghindari Deck Out)
    my_deck_count = my_state.deckCount
    opp_deck_count = opp_state.deckCount
    deck_diff = my_deck_count - opp_deck_count
    
    # Perumusan bobot (Prizes adalah prioritas tertinggi)
    # Prize diff max 6, bobot 0.1 -> +/- 0.6
    # HP diff max ~2000, bobot 0.0001 -> +/- 0.2
    # Deck diff max 60, bobot 0.001 -> +/- 0.06
    # Total potential maksimal akan selalu berada di bawah 1.0
    potential = (prize_diff * 0.1) + (hp_diff * 0.0001) + (deck_diff * 0.001)
    
    return float(potential)


def calculate_step_reward(old_potential: float, new_potential: float, state, your_index: int) -> float:
    """
    Menghitung Total Reward dari suatu transisi State.
    Menggabungkan Step Penalty, Potential Shaping, dan Terminal Reward.
    """
    if state is None:
        return 0.0
        
    # 1. Base Step (Time/Action Penalty untuk efisiensi agar tidak mengulur waktu)
    r_step = -0.001
    
    # 2. Shaping (Perubahan Potensial)
    # Ini yang akan mencegah infinite loop farming point (Zero-Sum pada loop)
    r_shaping = new_potential - old_potential
    
    # 3. Terminal State Result (Zero-Sum Reward Terbesar)
    r_terminal = 0.0
    if state.result != -1: # Game sudah berakhir
        if state.result == your_index:
            r_terminal = 1.0    # Menang
        elif state.result == 2:
            r_terminal = -0.1   # Draw (dianggap merugikan)
        else:
            r_terminal = -1.0   # Kalah
            
    total_reward = r_step + r_shaping + r_terminal
    
    # Proteksi numerik (clip reward agar tidak menimbulkan gradien meledak)
    return float(np.clip(total_reward, -5.0, 5.0))
