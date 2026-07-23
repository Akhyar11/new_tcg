# Reward System v5 - Improvement Report

## 📊 Summary
Reward system telah di-upgrade dari **v4** → **v5** dengan integrasi **database.csv** untuk threat assessment yang lebih akurat dan board state evaluation yang data-driven.

---

## 🎯 Peningkatan Utama

### 1. **Card Database Integration**
- ✅ Lazy-load `CardDB` dari `deck_ga/card_db.py`
- ✅ Akses real card data: HP, damage, energy cost, retreat cost, stage
- ✅ Robust fallback mode jika CSV tidak tersedia
- ✅ Circular import prevention dengan lazy-loading

### 2. **Threat Assessment System**
Fungsi baru: `_get_threat_score(pokemon_obj, card_db) → [0.0, 1.0]`

Menghitung threat level berdasarkan:
```
Threat = 0.4 * (Damage/300)              # Damage output (40%)
       + 0.3 * (Efficiency)               # Energy efficiency (30%)
       + 0.2 * (Mobility)                 # Retreat mobility (20%)
       + 0.15 * (Stage2 bonus)           # Evolutionary advantage (15%)
       + 0.1 * (EX bonus)                # Special rules (10%)
```

**Apa yang dipertimbangkan:**
- Max damage dari semua attacks
- Energy cost efficiency (damage-per-energy ratio)
- Retreat cost (mobilitas Pokemon)
- Evolutionary stage (Basic < Stage 1 < Stage 2)
- Special rules (ex, Tera, etc.)

### 3. **Board Quality Evaluation**
Fungsi baru: `_get_board_quality(player_state) → [0.0, 1.0]`

Evaluasi kualitas board:
```
Board Quality = 0.6 * (Threat Level) + 0.4 * (HP Abundance)
```

**Komponen:**
- Active Pokemon: 2x weight (paling penting)
- Bench Pokemon: 1x weight
- Total HP normalize ke 500 HP (full board)

### 4. **Improved Potential Calculation**

**SEBELUM (v4):**
```python
potential = (prize_diff * 0.1) + \
            (hp_ratio_diff * 0.015) + \
            (poke_count_diff * 0.002) + \
            (energy_diff * 0.0005) + \
            (deck_diff * 0.0002)
```

**SESUDAH (v5):**
```python
potential = (prize_diff * 0.15) + \              # ↑ 50% lebih penting
            (board_quality_diff * 0.12) + \      # NEW: Threat assessment
            (hp_ratio_diff * 0.08) + \           # Better tuning
            (poke_count_diff * 0.05) + \         # Better tuning
            (energy_diff * 0.03) + \             # Better tuning
            (deck_diff * 0.0002)
```

**Weight Perubahan:**
- Prize Difference: 0.10 → 0.15 (+50% weight - lebih penting)
- HP Ratio: 0.015 → 0.08 (+433% - better board representation)
- Pokemon Count: 0.002 → 0.05 (+2400% - better board representation)
- Energy: 0.0005 → 0.03 (+5900% - energy advantage lebih valuable)
- **NEW: Board Quality: 0.12** (threat assessment dengan card data)

---

## 📈 Impact pada Training

### Signaling yang Lebih Baik
| Scenario | Impact |
|----------|--------|
| Mengendalikan Pokemon ex dengan high damage | ↑ Reward: Threat assessment |
| Energy attach yang efisien | ↑ Reward: Better energy weight |
| Evolusi ke stage 2 | ↑ Reward: Evolutionary bonus |
| Retreat yang strategic | ↑ Reward: Mobility consideration |
| Menyerang dengan low energy cost | ↑ Reward: Efficiency scoring |

### Mencegah Exploit
- ✅ Zero-sum terminal reward: Farming bernilai 0
- ✅ Time penalty: -0.001 per step (encourage speed)
- ✅ Premature end penalty: -0.01 (anti-stalling)
- ✅ Clipping [-5, 5]: Numerical stability

---

## 🔧 Implementasi Detail

### A. Threat Score Calculation
```python
def _get_threat_score(pokemon_obj, card_db) -> float:
    # Lookup Pokemon card data from CSV
    card_data = card_db.by_id(pokemon_obj.id)
    
    threat = 0.0
    threat += min(1.0, max_dmg / 300.0) * 0.4      # Damage output
    threat += min(1.0, efficiency) * 0.3            # Energy efficiency
    threat += max(0.0, 1.0 - retreat/4.0) * 0.2   # Mobility
    
    if is_stage2: threat += 0.15
    if is_stage1: threat += 0.075
    if is_ex: threat += 0.1
    
    return clip(threat, 0.0, 1.0)
```

### B. Board Quality Calculation
```python
def _get_board_quality(player_state) -> float:
    # Active Pokemon: 2x weight (main threat)
    # Bench Pokemon: 1x weight (future threat)
    # Sum all threat scores, normalize by Pokemon count
    
    threat_component = threat_sum / max(1, count + 2)
    hp_component = min(1.0, total_hp / 500.0)
    
    return threat_component * 0.6 + hp_component * 0.4
```

### C. Fallback Mode
Jika CSV tidak ditemukan:
- CardDB = None
- Threat score = 0.0
- System masih berfungsi normal (tanpa threat data)
- Bonus dari evolutionary stage tetap ada

---

## 📝 File yang Dimodifikasi

### `/tcg_core/reward.py`
- ✅ Added: `_get_card_db()` - Lazy CardDB loader
- ✅ Added: `_get_threat_score()` - Threat assessment
- ✅ Added: `_get_board_quality()` - Board state evaluation
- ✅ Modified: `calculate_potential()` - Improved weights & card data
- ✅ Modified: `calculate_step_reward()` - Better documentation

### `/tcg_core/test_reward_v5.py` (NEW)
- Test card database loading
- Test threat scoring
- Test potential calculation
- Test reward calculation
- Validation script

---

## 🚀 Usage

### Training Loop (No Changes Required)
```python
from tcg_core.reward import calculate_step_reward

# Sistem akan otomatis load CSV saat diperlukan
reward = calculate_step_reward(old_state, new_state, player_index=0)
```

### Debug / Inspection
```python
from tcg_core.reward import _get_card_db, _get_threat_score

card_db = _get_card_db()  # Load database
card = card_db.by_id(30)  # Get Magcargo ex
print(f"Max damage: {card.max_damage}")
print(f"Energy cost: {card.total_energy_cost}")
print(f"Is EX: {card.is_ex}")
```

---

## ✅ Validation Results

Test suite: **PASSED** ✅

```
✓ CardDB loading (graceful fallback)
✓ Threat scoring (returns [0.0, 1.0])
✓ Potential calculation (returns [-1.0, 1.0])
✓ Reward calculation (returns [-5.0, 5.0])
✓ Victory/Loss/Draw logic
✓ Numerical stability
```

---

## 🔮 Future Improvements (Optional)

1. **Type Advantage Scoring** - Bonus jika weak to opponent, resistance
2. **Ability Parsing** - Extract ability text dari CSV untuk evaluation
3. **Hand Size Factor** - Include card draw advantage
4. **Status Effect Potential** - Poison, burn, paralyze, sleep scoring
5. **Stadium Effects** - Stadium card impact on board quality
6. **Tool/Attachment Value** - Pokemon tool bonus/malus

---

## 📌 Notes

- Kompatibel dengan existing training code (no breaking changes)
- Backward compatible: v4 calls masih work
- Performance impact: Negligible (lazy-load + caching)
- Memory impact: ~1-2 MB (CardDB loaded once)
- Fallback mode: Sistem tetap jalan tanpa database.csv

---

**Version:** 5  
**Date:** 23 Juli 2026  
**Status:** ✅ Production Ready
