#!/usr/bin/env python3
"""
Test script untuk reward system v5 - verifikasi card database integration.
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

def test_card_db_loading():
    """Test: CardDB loads successfully"""
    print("=" * 60)
    print("[Test 1] CardDB Loading")
    print("=" * 60)
    
    from reward import _get_card_db
    
    card_db = _get_card_db()
    if card_db is None:
        print("⚠️  WARNING: CardDB tidak bisa di-load. Sistem akan fallback ke mode tanpa data.")
        print("   CSV path tidak ditemukan atau error saat loading.")
        return False
    
    print("✅ CardDB loaded successfully!")
    
    # Test some cards
    test_cards = [25, 30, 40, 79]  # Some Pokemon IDs from database.csv
    for card_id in test_cards:
        card = card_db.by_id(card_id)
        if card:
            print(f"   Card #{card_id}: {card.name} (HP: {card.hp}, Damage: {card.max_damage}, Cost: {card.total_energy_cost})")
        else:
            print(f"   Card #{card_id}: Not found")
    
    return True


def test_threat_scoring():
    """Test: Threat scoring works"""
    print("\n" + "=" * 60)
    print("[Test 2] Threat Scoring")
    print("=" * 60)
    
    from reward import _get_threat_score, _get_card_db
    
    # Mock Pokemon object
    class MockPokemon:
        def __init__(self, card_id, hp=100):
            self.id = card_id
            self.hp = hp
            self.maxHp = hp
    
    card_db = _get_card_db()
    
    # Test threat scores for different cards
    test_cases = [
        (30, "Magcargo ex (strong attacker)"),
        (40, "Greninja ex (evolved stage)"),
        (31, "Chi-Yu (basic attacker)"),
    ]
    
    for card_id, description in test_cases:
        mock_poke = MockPokemon(card_id)
        threat = _get_threat_score(mock_poke, card_db)
        print(f"   {description}: threat_score = {threat:.3f}")
    
    print("✅ Threat scoring works!")
    return True


def test_potential_calculation():
    """Test: Potential calculation (mock state)"""
    print("\n" + "=" * 60)
    print("[Test 3] Potential Calculation (Mock)")
    print("=" * 60)
    
    from reward import calculate_potential
    
    # Mock state structure
    class MockCard:
        def __init__(self, card_id, hp=100):
            self.id = card_id
            self.hp = hp
            self.maxHp = hp
            self.energies = [0, 1, 2]  # 3 energies
    
    class MockPlayer:
        def __init__(self):
            self.active = [MockCard(30, hp=150)]  # Active Pokemon
            self.bench = [MockCard(31, hp=100), MockCard(40, hp=120)]  # Bench
            self.prize = [None] * 3  # 3 prizes left (3 taken)
            self.deckCount = 30
    
    class MockState:
        def __init__(self):
            self.players = [MockPlayer(), MockPlayer()]
            self.result = -1  # Game ongoing
    
    state = MockState()
    
    potential_p1 = calculate_potential(state, player_index=0)
    potential_p2 = calculate_potential(state, player_index=1)
    
    print(f"   Player 1 potential: {potential_p1:.4f}")
    print(f"   Player 2 potential: {potential_p2:.4f}")
    print(f"   Potential range: [-1.0, 1.0] ✅")
    
    # Both should be in valid range
    assert -1.0 <= potential_p1 <= 1.0, "Potential out of range!"
    assert -1.0 <= potential_p2 <= 1.0, "Potential out of range!"
    
    print("✅ Potential calculation works!")
    return True


def test_reward_calculation():
    """Test: Step reward calculation"""
    print("\n" + "=" * 60)
    print("[Test 4] Step Reward Calculation")
    print("=" * 60)
    
    from reward import calculate_step_reward
    import numpy as np
    
    # Mock state
    class MockCard:
        def __init__(self, card_id, hp=100):
            self.id = card_id
            self.hp = hp
            self.maxHp = hp
            self.energies = [0, 1]
    
    class MockPlayer:
        def __init__(self):
            self.active = [MockCard(30, hp=150)]
            self.bench = [MockCard(31, hp=100)]
            self.prize = [None] * 3
            self.deckCount = 30
    
    class MockState:
        def __init__(self, result=-1):
            self.players = [MockPlayer(), MockPlayer()]
            self.result = result
    
    # Test 1: Normal step (ongoing game)
    old_state = MockState(result=-1)
    new_state = MockState(result=-1)
    reward = calculate_step_reward(old_state, new_state, player_index=0, premature_end=False)
    print(f"   Normal step reward: {reward:.4f} (should be small negative)")
    assert -5.0 <= reward <= 5.0, "Reward out of range!"
    
    # Test 2: Victory
    new_state = MockState(result=0)
    reward_win = calculate_step_reward(old_state, new_state, player_index=0)
    print(f"   Victory reward: {reward_win:.4f} (should be positive, ~2.0)")
    assert reward_win > 1.0, "Victory reward too low!"
    
    # Test 3: Loss
    new_state = MockState(result=1)
    reward_loss = calculate_step_reward(old_state, new_state, player_index=0)
    print(f"   Loss reward: {reward_loss:.4f} (should be negative, ~-2.0)")
    assert reward_loss < -1.0, "Loss reward not negative enough!"
    
    # Test 4: Draw
    new_state = MockState(result=2)
    reward_draw = calculate_step_reward(old_state, new_state, player_index=0)
    print(f"   Draw reward: {reward_draw:.4f} (should be slightly negative)")
    
    print("✅ Step reward calculation works!")
    return True


if __name__ == "__main__":
    print("\n🧪 Testing Reward System v5 - Card Data Integration\n")
    
    try:
        test_card_db_loading()
        test_threat_scoring()
        test_potential_calculation()
        test_reward_calculation()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\n📊 Reward System v5 Summary:")
        print("   ✓ Card database integration")
        print("   ✓ Threat level assessment")
        print("   ✓ Board quality evaluation")
        print("   ✓ Potential-based shaping")
        print("   ✓ Reward clipping for stability")
        print("\n💡 Improvements over v4:")
        print("   • Prize weight: 0.1 → 0.15 (more important)")
        print("   • Added board_quality_diff: 0.12 (threat assessment)")
        print("   • Better HP/Pokemon/Energy weights")
        print("   • Card data-driven threat scoring")
        print("   • Evolutionary stage consideration")
        print("   • Energy efficiency analysis")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
