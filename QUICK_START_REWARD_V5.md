"""
Quick Start Guide - Reward System v5
====================================
"""

# 📌 PERUBAHAN RINGKAS

## Apa yang Diperbaiki?

Reward system v5 sekarang menggunakan data real dari database.csv untuk:
1. ✅ Threat Assessment - Hitung seberapa dangerous opponent Pokemon
2. ✅ Board Quality Evaluation - Score kualitas board lebih akurat
3. ✅ Energy Efficiency - Appreciate moves dengan cost-to-damage ratio bagus
4. ✅ Evolutionary Stage - Prefer evolved Pokemon (Stage 2 > Stage 1 > Basic)

---

## 🔄 PERUBAHAN WEIGHT DISTRIBUTION

### SEBELUM (v4):
```
Prize Diff:    × 0.1    (weak!)
HP Ratio:      × 0.015  (very weak)
Poke Count:    × 0.002  (negligible)
Energy:        × 0.0005 (negligible)
Deck:          × 0.0002 (minimal)
```

### SESUDAH (v5):
```
Prize Diff:       × 0.15      ↑ +50%
Board Quality:    × 0.12      ← NEW (threat assessment)
HP Ratio:         × 0.08      ↑ +433%
Poke Count:       × 0.05      ↑ +2400%
Energy:           × 0.03      ↑ +5900%
Deck:             × 0.0002    (sama)
```

---

## 🎮 PRAKTIK PENGGUNAAN

### 1. Training Loop (NO CHANGES NEEDED!)
```python
from tcg_core.reward import calculate_step_reward

# Sistem otomatis load database.csv pada saat dibutuhkan
reward = calculate_step_reward(old_state, new_state, player_index=0)
```

### 2. Debugging/Inspection
```python
from tcg_core.reward import _get_card_db, _get_threat_score

# Load database
card_db = _get_card_db()

# Inspect card
card = card_db.by_id(30)  # Magcargo ex
print(f"Name: {card.name}")
print(f"HP: {card.hp}")
print(f"Max Damage: {card.max_damage}")
print(f"Energy Cost: {card.total_energy_cost}")
print(f"Is EX: {card.is_ex}")
print(f"Stage: {card.stage}")

# Calculate threat
threat = _get_threat_score(mock_pokemon, card_db)
print(f"Threat Score: {threat:.3f}")
```

---

## ⚙️ TECHNICAL DETAILS

### New Functions

#### `_get_card_db() → Optional[CardDB]`
Lazy-loads card database dengan robust fallback.
- Return: CardDB instance atau None jika gagal
- Fallback: Sistem tetap berfungsi tanpa data

#### `_get_threat_score(pokemon_obj, card_db) → float [0.0, 1.0]`
Hitung threat level Pokemon.
- Damage output (40%)
- Energy efficiency (30%)
- Retreat mobility (20%)
- Evolutionary bonus (15%)
- Special rules bonus (10%)

#### `_get_board_quality(player_state) → float [0.0, 1.0]`
Hitung overall board quality.
- Active Pokemon: 2x weight
- Bench Pokemon: 1x weight
- HP abundance: normalisasi 500 HP

---

## 🧪 TESTING

Run validation:
```bash
cd /home/akhyar/Dokumen/Code/python/new_tcg
python tcg_core/test_reward_v5.py
```

Expected output:
```
✅ ALL TESTS PASSED!
 ✓ Card database integration
 ✓ Threat level assessment
 ✓ Board quality evaluation
 ✓ Potential-based shaping
 ✓ Reward clipping for stability
```

---

## 📊 EXPECTED TRAINING IMPACT

### Agent akan lebih prefer:
1. **High-damage Pokemon** (e.g., ex, Stage 2)
2. **Energy-efficient attacks** (high dmg, low cost)
3. **Mobile Pokemon** (low retreat cost)
4. **Strategic evolution** (prioritize stage 2)
5. **Fast decision-making** (time penalty)

### Agent akan avoid:
1. **Farming/looping** (zero-sum reward = 0 value)
2. **Wasting time** (step penalty = -0.001)
3. **Stalling** (premature end penalty = -0.01)
4. **Unnecessary bench Pokemon** (low contribution)

---

## 🔐 BACKWARD COMPATIBILITY

✅ **100% Compatible**
- Existing training code: NO CHANGES needed
- v4 imports: Still work (will call v5 internally)
- Fallback mode: Works without database.csv
- API: No breaking changes

---

## 📁 FILES MODIFIED

1. `/tcg_core/reward.py` - Main implementation
2. `/tcg_core/test_reward_v5.py` - Validation tests
3. `REWARD_V5_IMPROVEMENTS.md` - Detailed documentation

---

## 🚀 NEXT STEPS (OPTIONAL)

1. **Monitor Training** - Check if convergence improves
2. **Weight Tuning** - Adjust multipliers based on results
3. **Threat Factors** - Add type advantage, abilities, etc.
4. **Hand Advantage** - Include card draw advantage
5. **Status Effects** - Poison, burn, paralysis scoring

---

## ❓ TROUBLESHOOTING

### Q: CardDB tidak load?
A: Normal! Sistem fallback. Reward tetap berfungsi, hanya tanpa threat data.

### Q: Import error?
A: Pastikan running dari project root:
```bash
cd /home/akhyar/Dokumen/Code/python/new_tcg
python your_script.py
```

### Q: Bagaimana jika CSV berubah?
A: CardDB cache automatic reload on next import.

---

## 📝 VERSION INFO

- **Version:** v5
- **Date:** 23 Juli 2026
- **Status:** ✅ Production Ready
- **Breaking Changes:** None
- **Performance Impact:** Negligible

---

**READY TO USE!** 🎮✨
