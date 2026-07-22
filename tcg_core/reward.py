"""
Reward System — PPO training signals untuk Pokémon TCG.

v4 — True Potential-Based Reward Shaping
======================================
Menggunakan Zero-Sum Terminal Reward dan Potential-Based Reward Shaping.
Segala bentuk eksploitasi (farming/looping) bernilai matematis 0.

Potential (Phi) Formula:
- Prize Difference (Weight: 0.1)
- Board HP Difference (Weight: 0.0001)
- Deck Count Difference (Weight: 0.001)

Reward Formula:
R = (Phi_new - Phi_old) + R_step + R_terminal
"""
import numpy as np

def calculate_potential(state, player_index: int) -> float:
    if state is None:
        return 0.0
        
    my_state = state.players[player_index]
    opp_state = state.players[1 - player_index]
    
    # 1. Prize Card Potential (Score bounds roughly +/- 0.6)
    my_prize_taken = 6 - len(my_state.prize)
    opp_prize_taken = 6 - len(opp_state.prize)
    prize_diff = my_prize_taken - opp_prize_taken
    
    # 2. Board HP Advantage
    def get_total_hp(player_state):
        active_hp = sum(p.hp for p in player_state.active if p)
        bench_hp = sum(p.hp for p in player_state.bench if p)
        return active_hp + bench_hp
        
    hp_diff = get_total_hp(my_state) - get_total_hp(opp_state)
    
    # 3. Deck Count (Prevent deck-out)
    deck_diff = my_state.deckCount - opp_state.deckCount
    
    potential = (prize_diff * 0.1) + (hp_diff * 0.0001) + (deck_diff * 0.001)
    return potential

def calculate_step_reward(old_state, new_state, player_index: int, end_reason: int = 0, premature_end: bool = False) -> float:
    """
    Menghitung reward per step dengan Potential-Based Shaping.
    """
    if new_state is None:
        return -2.0 # Invalid state crash

    # Hitung Shaping Reward
    old_potential = calculate_potential(old_state, player_index)
    new_potential = calculate_potential(new_state, player_index)
    r_shaping = new_potential - old_potential
    
    # Strict Time Penalty (Setiap klik / action step dikenakan penalti konstan)
    r_step = -0.001
    
    # Hukuman ekstra jika pass turn tanpa alasan (stalling)
    if premature_end:
        r_step -= 0.01

    # Terminal State Reward (Zero-Sum)
    r_terminal = 0.0
    if new_state.result != -1:
        if new_state.result == player_index:
            r_terminal = 2.0
        elif new_state.result == 2:
            r_terminal = -0.1  # Draw penalty sedikit
        else:
            r_terminal = -2.0
            
    total_reward = r_step + r_shaping + r_terminal
    
    # Clip extreme values (just in case)
    return float(np.clip(total_reward, -5.0, 5.0))
