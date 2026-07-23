"""
Reward System — PPO training signals untuk Pokémon TCG.

v4 — True Potential-Based Reward Shaping
======================================
Menggunakan Zero-Sum Terminal Reward dan Potential-Based Reward Shaping.
Segala bentuk eksploitasi (farming/looping) bernilai matematis 0.

Potential (Phi) Formula:
- Prize Difference (Weight: 0.1)
- Board Presence (HP Ratio: 0.015, Pokemon: 0.002, Energy: 0.0005)
- Deck Count Difference (Weight: 0.0002)

Reward Formula:
R = (gamma * Phi_new - Phi_old) + R_step + R_terminal
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
    
    # 2. Board Presence (HP Ratio, Pokemon Count, Energy Count)
    def get_board_stats(player_state):
        hp_ratio_sum = 0.0
        pokemon_count = 0
        energy_count = 0
        
        for p in player_state.active + player_state.bench:
            if p:
                pokemon_count += 1
                # get max_hp safely and clamp ratio to [0, 1]
                max_hp = getattr(p, 'maxHp', 0)
                safe_max_hp = max(max_hp, p.hp, 1)
                hp_ratio_sum += min(1.0, p.hp / safe_max_hp)
                
                # get energy (saturate at 4 to prevent over-attach hacking)
                energies = getattr(p, 'energies', [])
                energy_count += min(len(energies), 4)
                
        return hp_ratio_sum, pokemon_count, energy_count
        
    my_hp_ratio, my_poke_count, my_energy_count = get_board_stats(my_state)
    opp_hp_ratio, opp_poke_count, opp_energy_count = get_board_stats(opp_state)
    
    hp_ratio_diff = my_hp_ratio - opp_hp_ratio
    poke_count_diff = my_poke_count - opp_poke_count
    energy_diff = my_energy_count - opp_energy_count
    
    # 3. Deck Count (Prevent deck-out)
    deck_diff = my_state.deckCount - opp_state.deckCount
    
    potential = (prize_diff * 0.1) + \
                (hp_ratio_diff * 0.015) + \
                (poke_count_diff * 0.002) + \
                (energy_diff * 0.0005) + \
                (deck_diff * 0.0002)
    return float(np.clip(potential, -1.0, 1.0))

def calculate_step_reward(old_state, new_state, player_index: int, end_reason: int = 0, premature_end: bool = False) -> float:
    """
    Menghitung reward per step dengan Potential-Based Shaping.
    """
    if new_state is None:
        return -2.0 # Invalid state crash

    # Hitung Shaping Reward
    old_potential = calculate_potential(old_state, player_index)
    new_potential = calculate_potential(new_state, player_index)
    gamma = 0.99
    r_shaping = (gamma * new_potential) - old_potential
    
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
