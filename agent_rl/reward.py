"""
Reward System — PPO training signals untuk Pokémon TCG.

v3 — Convergence-Grade Reward Design
======================================
Prinsip desain agar konvergen di 5-10M timesteps:

1. TERMINAL : INTERMEDIATE ≈ 1:1
   Terminal ±1.5, intermediate max ~1.0 per step.
   Gradient tidak didominasi 1-2 steps terakhir.

2. SYMMETRIC DECK-OUT
   Deck-out = 0 untuk PEMENANG MAUPUN PECUNDANG.
   Model tidak belajar stalling, juga tidak dihukum
   karena situasi di luar kendali.

3. STRATEGIC REWARDS
   - Bench building (fundamental TCG)
   - Hand-size awareness (draw engine)
   - Energy-type matching (energy management)
   - Retreat cost awareness

4. ANTI-HACKING (dipertahankan dari v2)
   - Net damage (dealt - received)
   - Serial tracking untuk evolve
   - State-based verification
   - Diminishing returns per event category

Reward Budget per Game (~50 steps):
  Step penalty:     -1.5  s.d.  -3.0
  Damage dealt:      0.0  s.d.  +2.5
  Damage received:   0.0  s.d.  -1.0
  Prize taken:       0.0  s.d.  +3.0
  Evolve:            0.0  s.d.  +0.9
  Bench building:    0.0  s.d.  +0.5
  Energy attach:     0.0  s.d.  +0.3
  Supporter/Item:    0.0  s.d.  +0.2
  Terminal:         -1.5  s.d.  +1.5
  ─────────────────────────────────
  Total:            -2.0  s.d.  +6.0
"""
import numpy as np
from cg.api import all_card_data, CardType, LogType

# Counter global per-game untuk diminishing returns (direset via reset_trackers())
_event_counters = {}

# Cache card & attack database per-worker
_CARD_DB = None
_ATTACK_DB = None

def _get_card_db():
    global _CARD_DB
    if _CARD_DB is None:
        _CARD_DB = {c.cardId: c for c in all_card_data()}
    return _CARD_DB


def _get_attack_db():
    global _ATTACK_DB
    if _ATTACK_DB is None:
        from cg.api import all_attack
        _ATTACK_DB = {a.attackId: a for a in all_attack()}
    return _ATTACK_DB


def reset_trackers():
    """Reset event counters untuk game baru."""
    _event_counters.clear()
    _event_counters['ready_serials'] = set()


def _increment_counter(key: str) -> int:
    """Increment counter dan return nilai sebelum increment."""
    val = _event_counters.get(key, 0)
    _event_counters[key] = val + 1
    return val


def detect_events(old_state, new_state, player_index: int, logs: list = None) -> dict:
    """
    Mendeteksi event apa yang terjadi dalam satu step.
    Berbasis state comparison, diverifikasi dengan logs.
    """
    events = {}
    if old_state is None or new_state is None:
        return events

    my_old = old_state.players[player_index]
    my_new = new_state.players[player_index]
    opp_index = 1 - player_index
    opp_old = old_state.players[opp_index]
    opp_new = new_state.players[opp_index]

    # 1. Energy Attach
    old_energy = sum(len(p.energies) for p in my_old.active if p) + \
                 sum(len(p.energies) for p in my_old.bench if p)
    new_energy = sum(len(p.energies) for p in my_new.active if p) + \
                 sum(len(p.energies) for p in my_new.bench if p)
    if new_energy > old_energy:
        events['energy_attached'] = new_energy - old_energy

    # 2. Prize Taken
    old_prize = len(my_old.prize)
    new_prize = len(my_new.prize)
    if new_prize < old_prize:
        events['prize_taken'] = old_prize - new_prize
        events['ko'] = True

    # 3. NET Damage
    old_opp_active = opp_old.active
    new_opp_active = opp_new.active
    old_my_active = my_old.active
    new_my_active = my_new.active

    damage_dealt = 0
    damage_received = 0
    has_hp_logs = False
    if logs:
        for log in logs:
            if log.type == LogType.HP_CHANGE:
                val = log.value if log.value is not None else 0
                if val < 0:
                    has_hp_logs = True
                    dmg = -val
                    if log.playerIndex == opp_index:
                        damage_dealt += dmg
                    elif log.playerIndex == player_index:
                        damage_received += dmg

    if not has_hp_logs:
        if old_opp_active and old_opp_active[0] and new_opp_active and new_opp_active[0]:
            if old_opp_active[0].serial == new_opp_active[0].serial:
                hp_loss = old_opp_active[0].hp - new_opp_active[0].hp
                if hp_loss > 0:
                    damage_dealt = hp_loss

        if old_my_active and old_my_active[0] and new_my_active and new_my_active[0]:
            if old_my_active[0].serial == new_my_active[0].serial:
                hp_loss_self = old_my_active[0].hp - new_my_active[0].hp
                if hp_loss_self > 0:
                    damage_received = hp_loss_self

    net_damage = damage_dealt - damage_received
    if net_damage > 0:
        events['net_damage'] = net_damage
    elif damage_received > 0:
        events['damage_received'] = damage_received

    # 4. Evolusi Active (serial tracking — verify bukan switch/retreat)
    if old_my_active and old_my_active[0] and new_my_active and new_my_active[0]:
        old_id = old_my_active[0].id
        new_id = new_my_active[0].id
        if old_id != new_id:
            old_serial = old_my_active[0].serial
            old_on_bench = any(p.serial == old_serial for p in my_new.bench if p)
            if not old_on_bench:
                old_bench_n = sum(1 for p in my_old.bench if p)
                new_bench_n = sum(1 for p in my_new.bench if p)
                if new_bench_n >= old_bench_n:
                    events['evolved'] = True

    # 5. Evolusi Bench
    for old_b in [p for p in my_old.bench if p]:
        old_serial_b = old_b.serial
        still_on_field = any(
            (p and p.serial == old_serial_b)
            for p in list(my_new.bench) + list(my_new.active)
        )
        if not still_on_field:
            old_bench_n = sum(1 for p in my_old.bench if p)
            new_bench_n = sum(1 for p in my_new.bench if p)
            if new_bench_n >= old_bench_n:
                events['bench_evolved'] = True
                break

    # 6. Bench Building — Pokemon baru di bench
    old_bench_count = sum(1 for p in my_old.bench if p and p.hp > 0)
    new_bench_count = sum(1 for p in my_new.bench if p and p.hp > 0)
    if new_bench_count > old_bench_count:
        events['bench_built'] = new_bench_count - old_bench_count

    # 7. Hand Size — estimasi dari bench/active count changes
    # Tidak bisa langsung, tapi bisa dideteksi dari log
    if logs:
        card_db = _get_card_db()
        for log in logs:
            if log.type != LogType.PLAY or log.playerIndex != player_index:
                continue
            if log.cardId is None:
                continue
            card = card_db.get(log.cardId)
            if card is None:
                continue
            if card.cardType == CardType.SUPPORTER:
                events['supporter_played'] = True
            elif card.cardType == CardType.ITEM:
                events['item_played'] = True

    # Ensure ready_serials exists
    if 'ready_serials' not in _event_counters:
        _event_counters['ready_serials'] = set()
    ready_serials = _event_counters['ready_serials']

    # 8. Battle Ready Milestone (Kesiapan Tempur)
    card_db = _get_card_db()
    attack_db = _get_attack_db()
    
    my_pokemon = [p for p in list(my_new.bench) + list(my_new.active) if p]
    for p in my_pokemon:
        p_id = p.id
        p_serial = p.serial
        card_data = card_db.get(p_id)
        if card_data and getattr(card_data, 'attacks', None):
            total_cost = 0
            for atk_id in card_data.attacks:
                atk = attack_db.get(atk_id)
                if atk and getattr(atk, 'energies', None):
                    if len(atk.energies) > total_cost:
                        total_cost = len(atk.energies)
            
            if total_cost > 0 and len(p.energies) >= total_cost:
                if p_serial not in ready_serials:
                    ready_serials.add(p_serial)
                    events['battle_ready'] = events.get('battle_ready', 0) + 1

    # 9. Strategic / Normal Retreat
    if old_my_active and old_my_active[0] and new_my_active and new_my_active[0]:
        if not old_state.retreated and new_state.retreated:
            old_serial = old_my_active[0].serial
            retreated_bench = [p for p in my_new.bench if p and p.serial == old_serial]
            if retreated_bench:
                hp_ratio = old_my_active[0].hp / old_my_active[0].maxHp
                if hp_ratio < 0.5:
                    events['strategic_retreat'] = True
                else:
                    events['normal_retreat'] = True

    return events


def calculate_step_reward(new_state, player_index: int, events: dict = None, end_reason: int = 0) -> float:
    """
    Reward dengan skala seimbang untuk konvergensi PPO.
    """
    if new_state is None:
        return 0.0

    turn = new_state.turn
    my_new = new_state.players[player_index]

    # ── 1. Step penalty — exponential, stall semakin mahal ──
    # Turn 1: -0.03, Turn 50: -0.05, Turn 100: -0.08, Turn 200: -0.12
    r_step = -0.03 * (1.15 ** (turn / 50.0))

    # ── 2. Intermediate rewards ──
    r_event = 0.0

    if events:
        # Bench building — minor breadcrumb
        if events.get('bench_built', 0) > 0:
            n = _increment_counter('bench')
            decay = 0.50 ** n
            r_event += 0.02 * events['bench_built'] * decay

        # Energy attach — minor breadcrumb
        if events.get('energy_attached', 0) > 0:
            n = _increment_counter('energy')
            decay = 0.50 ** n
            r_energy = 0.03 * events['energy_attached'] * decay
            r_event += r_energy

        # Evolution (active)
        if events.get('evolved'):
            n = _increment_counter('evolve')
            decay = 0.50 ** n
            r_event += 0.05 * decay

        # Evolution (bench)
        if events.get('bench_evolved'):
            n = _increment_counter('bench_evolve')
            decay = 0.50 ** n
            r_event += 0.03 * decay

        # Supporter
        if events.get('supporter_played'):
            n = _increment_counter('supporter')
            decay = 0.50 ** n
            r_event += 0.02 * decay

        # Item
        if events.get('item_played'):
            n = _increment_counter('item')
            decay = 0.50 ** n
            r_event += 0.02 * decay

        # Battle Ready Milestone (Kesiapan Tempur)
        if events.get('battle_ready', 0) > 0:
            n = _increment_counter('battle_ready')
            decay = 0.50 ** n
            r_event += 0.05 * events['battle_ready'] * decay

        # Strategic / Normal Retreat
        if events.get('strategic_retreat'):
            n = _increment_counter('retreat')
            decay = 0.50 ** n
            r_event += 0.05 * decay
        elif events.get('normal_retreat'):
            n = _increment_counter('retreat')
            decay = 0.50 ** n
            r_event += 0.00 * decay

        # Net damage — Primary source of intermediate reward
        if events.get('net_damage', 0) > 0:
            # 10 damage = +0.04. 100 damage = +0.40. Cap at +1.0 per step
            r_damage = min((events['net_damage'] / 100.0) * 0.40, 1.0)
            r_event += r_damage

        # Damage received
        if events.get('damage_received', 0) > 0:
            # 10 damage = -0.02. 100 damage = -0.20
            r_penalty = min((events['damage_received'] / 100.0) * 0.20, 1.0)
            r_event -= r_penalty

        # Prize taken — Large intermediate milestone
        if events.get('prize_taken', 0) > 0:
            n_prizes = events['prize_taken']
            if n_prizes >= 2:
                r_event += 1.20
            else:
                r_event += 0.80

        # KO without prize
        if events.get('ko') and not events.get('prize_taken'):
            r_event += 0.30

    # ── 3. Intra-Game Reward Annealing ──
    # Kurangi bobot intermediate reward seiring berkurangnya prize card (game mendekati akhir)
    prizes_left = len(my_new.prize) if my_new and my_new.prize else 6
    intermediate_scale = max(0.1, prizes_left / 6.0)
    
    # Juga kurangi seiring bertambahnya turn (late game)
    turn_scale = max(0.1, 1.0 - (turn / 100.0))
    
    # Kalikan r_event dengan faktor skala gabungan
    r_event = r_event * (intermediate_scale * turn_scale)

    # Cap intermediate per step
    r_event = np.clip(r_event, -1.5, 2.5)

    # ── 4. Terminal reward ──
    r_terminal = 0.0
    if new_state.result != -1:
        won = (new_state.result == player_index)
        lost = (new_state.result == (1 - player_index))
        draw = (new_state.result == 2)

        if draw:
            r_terminal = 0.0
        elif end_reason == 1:
            # Prize win/loss
            r_terminal = 2.0 if won else -2.0
        elif end_reason == 2:
            # Deck-out — 0 untuk KEDUA sisi (symmetric)
            r_terminal = 0.0
        elif end_reason in (3, 4):
            # NoActive / Effect
            r_terminal = 0.50 if won else -0.50
        else:
            # Fallback
            r_terminal = 1.0 if won else -1.0

    total = r_step + r_event + r_terminal
    return float(np.clip(total, -5.0, 5.0))
