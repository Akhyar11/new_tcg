"""
Feature Extractor — Ekstrak fitur numerik dari deck untuk surrogate model.

Menghasilkan vector dengan dimensi tetap (FIXED_DIM) yang bisa diprediksi
oleh surrogate model untuk memperkirakan win rate tanpa simulasi penuh.
"""
import numpy as np
from typing import Optional

from .card_db import CardDB, CardType

# Dimensi fitur tetap per deck
# 7 card types + 8 energy types + 6 stat features = 21 feature groups
FIXED_DIM = 64


def extract_deck_features(card_ids: list[int], db: CardDB) -> np.ndarray:
    """
    Ekstrak fitur numerik dari deck.

    Feature vector (64 dimensions):
    Index   Description
    ─────   ──────────────────────────────────────────
    0-6     Card type counts (Basic, S1, S2, Item, Supporter, Tool, Stadium)
    7-14    Energy type distribution
    15-18   Average stats (HP, retreat cost, max damage, energy cost)
    19      Number of evolution lines
    20-27   Energy curve (how many attacks at each cost: 0-7+)
    28-34   Trainer subtype counts
    35-42   Damage histogram bins
    43-50   HP histogram bins
    51-57   Evolution support (basic with evo, S1 without basic, etc.)
    58-62   Special flags (ace spec count, ex count, basic energy count)
    63      Deck size (always 60, sanity check)
    """
    features = np.zeros(FIXED_DIM, dtype=np.float32)

    cards = [db.by_id(cid) for cid in card_ids]
    cards = [c for c in cards if c]

    if not cards:
        return features

    # 0-6: Card type counts
    for c in cards:
        if c.stage == CardType.BASIC_POKEMON:
            features[0] += 1
        elif c.stage == CardType.STAGE1_POKEMON:
            features[1] += 1
        elif c.stage == CardType.STAGE2_POKEMON:
            features[2] += 1
        elif c.stage == CardType.ITEM:
            features[3] += 1
        elif c.stage == CardType.SUPPORTER:
            features[4] += 1
        elif c.stage == CardType.TOOL:
            features[5] += 1
        elif c.stage == CardType.STADIUM:
            features[6] += 1

    # Normalize by deck size
    features[0:7] /= max(len(cards), 1)

    # 7-14: Energy type distribution (from basic energy cards)
    for c in cards:
        if c.is_energy and c.stage == CardType.BASIC_ENERGY:
            sym = c.energy_type.strip("{}")
            type_idx = {"G": 0, "R": 1, "W": 2, "L": 3, "P": 4, "F": 5, "D": 6, "M": 7}.get(sym, -1)
            if type_idx >= 0:
                features[7 + type_idx] += 1

    # 15-18: Average stats for Pokemon
    pokemon = [c for c in cards if c.is_pokemon]
    if pokemon:
        features[15] = np.mean([p.hp for p in pokemon]) / 300.0
        features[16] = np.mean([p.retreat for p in pokemon]) / 5.0
        features[17] = np.mean([p.max_damage for p in pokemon]) / 300.0
        features[18] = np.mean([p.total_energy_cost for p in pokemon]) / 10.0

    # 19: Number of evolution lines (estimated from unique basic Pokemon names)
    basic_names = set()
    for c in pokemon:
        if c.is_basic:
            basic_names.add(c.name)
    features[19] = len(basic_names) / 6.0

    # 20-27: Energy curve (attack cost distribution)
    cost_bins = [0] * 8
    for p in pokemon:
        cost = p.total_energy_cost
        bin_idx = min(cost, 7)
        cost_bins[bin_idx] += 1
    for i in range(8):
        features[20 + i] = cost_bins[i] / max(len(pokemon), 1)

    # 28-34: Trainer subtype counts
    for c in cards:
        if c.is_trainer:
            if c.stage == CardType.ITEM:
                features[28] += 1
            elif c.stage == CardType.SUPPORTER:
                features[29] += 1
            elif c.stage == CardType.TOOL:
                features[30] += 1
            elif c.stage == CardType.STADIUM:
                features[31] += 1
    features[28:32] /= max(len(cards), 1)

    # 35-42: Damage histogram
    dmg_bins = [0] * 8
    for p in pokemon:
        dmg = p.max_damage
        bin_idx = min(dmg // 50, 7)
        dmg_bins[bin_idx] += 1
    for i in range(8):
        features[35 + i] = dmg_bins[i] / max(len(pokemon), 1)

    # 43-50: HP histogram
    hp_bins = [0] * 8
    for p in pokemon:
        hp_bin = min(p.hp // 40, 7)
        hp_bins[hp_bin] += 1
    for i in range(8):
        features[43 + i] = hp_bins[i] / max(len(pokemon), 1)

    # 51-57: Evolution support
    # Stage1 count vs Basic count ratio
    if features[0] > 0:
        features[51] = features[1] / features[0]  # S1/Basic ratio
    features[52] = features[2] / max(features[1], 1)  # S2/S1 ratio
    # Basic Pokemon with no evolution option
    s1_names = set()
    for c in cards:
        if c.is_stage1:
            s1_names.add(c.prev_stage_name)
    basic_no_evo = sum(1 for c in pokemon if c.is_basic and c.name not in s1_names)
    features[53] = basic_no_evo / max(len(pokemon), 1)

    # 58-62: Special flags
    features[58] = sum(1 for c in cards if c.is_ace_spec)  # ACE SPEC count
    features[59] = sum(1 for c in cards if c.is_ex) / max(len(pokemon), 1)  # ex ratio
    features[60] = sum(1 for c in cards if c.is_energy and c.stage == CardType.BASIC_ENERGY) / 15.0  # basic energy count
    features[61] = sum(1 for c in cards if c.is_energy and c.stage == CardType.SPECIAL_ENERGY) / 5.0  # special energy
    features[62] = features[7] + features[8] + features[9] + features[10]  # total energy ratio

    # 63: Deck size (sanity check)
    features[63] = len(cards) / 60.0

    return features
