"""
Reward System — PPO training signals untuk Pokémon TCG.

v5 — True Potential-Based Reward Shaping with Card Data Integration
==================================================================
Menggunakan Zero-Sum Terminal Reward dan Potential-Based Reward Shaping.
Mengintegrasikan database.csv untuk threat assessment & board state evaluation.
Segala bentuk eksploitasi (farming/looping) bernilai matematis 0.

Potential (Phi) Formula:
- Prize Difference (Weight: 0.1)
- Board Threat & Quality (Damage output, energy efficiency, evolutionary stage)
- HP Ratio & Pokemon Count
- Energy Advantage
- Deck Count Difference (Weight: 0.0002)

Reward Formula:
R = (gamma * Phi_new - Phi_old) + R_step + R_terminal
"""
import numpy as np
import os
from pathlib import Path
from typing import Optional

# ─── Lazy-load CardDB ───
_CARD_DB = None

def _get_card_db():
    """Lazy-load CardDB untuk menghindari circular imports."""
    global _CARD_DB
    if _CARD_DB is None:
        try:
            # Try multiple import strategies
            CardDB = None
            try:
                from deck_ga.card_db import CardDB
            except ImportError:
                # If direct import fails, try adding parent to path
                import sys
                parent_dir = os.path.dirname(os.path.dirname(__file__))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from deck_ga.card_db import CardDB
            
            csv_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "cg", 
                "database.csv"
            )
            csv_path = os.path.abspath(csv_path)
            
            if os.path.exists(csv_path):
                _CARD_DB = CardDB(csv_path)
            else:
                raise FileNotFoundError(f"CSV not found at {csv_path}")
                
        except Exception as e:
            # Fallback: system still works but without card data
            _CARD_DB = False  # Mark as unavailable
    return _CARD_DB if _CARD_DB is not False else None


def _get_threat_score(pokemon_obj, card_db) -> float:
    """
    Hitung threat level dari satu Pokemon berdasarkan:
    - Max damage dari attacks
    - Energy cost efficiency
    - Retreat cost (mobilitas)
    - Evolutionary stage (evolved = higher threat)
    """
    if card_db is None or pokemon_obj is None:
        return 0.0
    
    try:
        card_data = card_db.by_id(pokemon_obj.id)
        if card_data is None:
            return 0.0
        
        threat = 0.0
        
        # 1. Damage Output (normalize to [0, 1] assuming max ~300 damage)
        max_dmg = card_data.max_damage
        threat += min(1.0, max_dmg / 300.0) * 0.4
        
        # 2. Energy Efficiency (lower cost = higher threat)
        # Ideal: high damage, low cost
        total_energy = card_data.total_energy_cost
        if total_energy > 0:
            efficiency = max_dmg / total_energy / 50.0  # normalize
            threat += min(1.0, efficiency) * 0.3
        
        # 3. Retreat Cost (lower = higher mobility threat)
        retreat_mobility = max(0.0, 1.0 - card_data.retreat / 4.0)
        threat += retreat_mobility * 0.2
        
        # 4. Evolutionary Stage (Stage 2 > Stage 1 > Basic)
        if card_data.is_stage2:
            threat += 0.15
        elif card_data.is_stage1:
            threat += 0.075
        
        # 5. Special rules (ex, Tera = higher threat potential)
        if card_data.is_ex:
            threat += 0.1
        
        return float(np.clip(threat, 0.0, 1.0))
    except Exception:
        return 0.0


def _get_board_quality(player_state, is_opponent: bool = False) -> float:
    """
    Hitung kualitas board berdasarkan:
    - Threat level dari active Pokemon
    - Bench Pokemon count & quality
    - Total HP available
    - Energy attachment level
    """
    card_db = _get_card_db()
    
    total_threat = 0.0
    pokemon_count = 0
    total_hp = 0.0
    active_threat = 0.0
    
    # Active Pokemon (most important)
    if player_state.active and len(player_state.active) > 0 and player_state.active[0]:
        active_poke = player_state.active[0]
        active_threat = _get_threat_score(active_poke, card_db) * 2.0  # 2x weight for active
        total_threat += active_threat
        total_hp += active_poke.hp
        pokemon_count += 1
    
    # Bench Pokemon
    for bench_poke in player_state.bench:
        if bench_poke:
            pokemon_count += 1
            threat = _get_threat_score(bench_poke, card_db)
            total_threat += threat
            total_hp += bench_poke.hp
    
    # Normalize: board quality is weighted threat + HP
    # Weight: 60% threat level, 40% HP abundance
    threat_component = total_threat / max(1, pokemon_count + 2)  # normalize
    hp_component = min(1.0, total_hp / 500.0)  # normalize to 500 HP as "full"
    
    board_quality = threat_component * 0.6 + hp_component * 0.4
    return float(np.clip(board_quality, 0.0, 1.0))


def calculate_potential(state, player_index: int) -> float:
    """
    Hitung potential state dengan card data integration.
    """
    if state is None:
        return 0.0
        
    my_state = state.players[player_index]
    opp_state = state.players[1 - player_index]
    
    # 1. Prize Card Potential (Score bounds roughly +/- 0.6)
    my_prize_taken = 6 - len(my_state.prize)
    opp_prize_taken = 6 - len(opp_state.prize)
    prize_diff = my_prize_taken - opp_prize_taken
    
    # 2. Board Quality & Threat Assessment (with card data)
    my_board_quality = _get_board_quality(my_state, is_opponent=False)
    opp_board_quality = _get_board_quality(opp_state, is_opponent=True)
    board_quality_diff = my_board_quality - opp_board_quality
    
    # 3. Traditional Board Stats (HP Ratio, Pokemon Count, Energy)
    def get_board_stats(player_state):
        hp_ratio_sum = 0.0
        pokemon_count = 0
        energy_count = 0
        
        for p in player_state.active + player_state.bench:
            if p:
                pokemon_count += 1
                max_hp = getattr(p, 'maxHp', 1)
                safe_max_hp = max(max_hp, 1)
                hp_ratio_sum += min(1.0, p.hp / safe_max_hp)
                
                energies = getattr(p, 'energies', [])
                energy_count += min(len(energies), 4)
                
        return hp_ratio_sum, pokemon_count, energy_count
        
    my_hp_ratio, my_poke_count, my_energy_count = get_board_stats(my_state)
    opp_hp_ratio, opp_poke_count, opp_energy_count = get_board_stats(opp_state)
    
    hp_ratio_diff = my_hp_ratio - opp_hp_ratio
    poke_count_diff = my_poke_count - opp_poke_count
    energy_diff = my_energy_count - opp_energy_count
    
    # 4. Deck Count (Prevent deck-out)
    deck_diff = my_state.deckCount - opp_state.deckCount
    
    # Combine potentials dengan improved weights
    potential = (prize_diff * 0.15) + \
                (board_quality_diff * 0.12) + \
                (hp_ratio_diff * 0.08) + \
                (poke_count_diff * 0.05) + \
                (energy_diff * 0.03) + \
                (deck_diff * 0.0002)
    
    return float(np.clip(potential, -1.0, 1.0))



def calculate_step_reward(old_state, new_state, player_index: int, end_reason: int = 0, premature_end: bool = False) -> float:
    """
    Menghitung reward per step dengan Potential-Based Shaping.
    
    Komponen reward:
    - r_shaping: Perubahan potential state (mendorong board improvement)
    - r_step: Time penalty (encourage efficiency)
    - r_terminal: Zero-sum win/loss reward
    
    Total reward di-clip ke [-5, 5] untuk numerical stability.
    """
    if new_state is None:
        return -2.0  # Invalid state crash

    # Hitung Shaping Reward dengan improved potential
    old_potential = calculate_potential(old_state, player_index)
    new_potential = calculate_potential(new_state, player_index)
    gamma = 0.99
    r_shaping = (gamma * new_potential) - old_potential
    
    # Strict Time Penalty (Setiap action step dikenakan penalti konstan)
    # Mendorong agent untuk bermain cepat dan decisive
    r_step = -0.001
    
    # Hukuman ekstra jika pass turn tanpa alasan (stalling)
    # atau premature ending (surrender)
    if premature_end:
        r_step -= 0.01

    # Terminal State Reward (Zero-Sum)
    # Win: +2.0 | Draw: -0.1 | Loss: -2.0
    r_terminal = 0.0
    if new_state.result != -1:  # -1 means game still ongoing
        if new_state.result == player_index:
            r_terminal = 2.0  # Victory!
        elif new_state.result == 2:
            r_terminal = -0.1  # Draw (slight penalty)
        else:
            r_terminal = -2.0  # Loss
            
    total_reward = r_step + r_shaping + r_terminal
    
    # Clip extreme values untuk numerical stability
    return float(np.clip(total_reward, -5.0, 5.0))
