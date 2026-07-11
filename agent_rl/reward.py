import numpy as np
from cg.api import all_card_data, CardType, LogType

# Counter global per-game untuk diminishing returns (direset via reset_trackers())
_event_counters = {}

# Cache card database per-worker (lazy-init, masing-masing worker fork punya sendiri)
_CARD_DB = None

def _get_card_db():
    global _CARD_DB
    if _CARD_DB is None:
        _CARD_DB = {c.cardId: c for c in all_card_data()}
    return _CARD_DB


def reset_trackers():
    """Reset event counters untuk game baru (dipanggil di auto-reset)."""
    _event_counters.clear()


def _increment_counter(key: str) -> int:
    """Increment counter dan return nilai sebelum increment."""
    val = _event_counters.get(key, 0)
    _event_counters[key] = val + 1
    return val


def detect_events(old_state, new_state, player_index: int, logs: list = None) -> dict:
    """
    Mendeteksi event apa yang terjadi dalam satu step dengan membandingkan
    state sebelum dan sesudah aksi dieksekusi, dilengkapi verifikasi dari logs.

    Anti-hacking:
    - Hanya mendeteksi perubahan NET (bukan gross)
    - Damage dicek dari state aktual (bisa diverifikasi)
    - Prize hanya terdeteksi jika prize beneran berkurang
    - Evolusi diverifikasi via serial tracking (bukan heuristic energi)

    Returns:
        dict: events yang terdeteksi
    """
    events = {}
    if old_state is None or new_state is None:
        return events

    # 1. Energy Attach: flag energyAttached engine + bukti jumlah energy naik
    if not old_state.energyAttached and new_state.energyAttached:
        my_old = old_state.players[player_index]
        my_new = new_state.players[player_index]
        # Verifikasi: total energi di board (active + bench) benar-benar naik
        old_energy_count = sum(len(p.energies) for p in my_old.active if p) + sum(len(p.energies) for p in my_old.bench if p)
        new_energy_count = sum(len(p.energies) for p in my_new.active if p) + sum(len(p.energies) for p in my_new.bench if p)
        if new_energy_count > old_energy_count:
            events['energy_attached'] = new_energy_count - old_energy_count

    my_old = old_state.players[player_index]
    my_new = new_state.players[player_index]

    # 2. Prize Taken
    old_prize_count = len(my_old.prize)
    new_prize_count = len(my_new.prize)
    if new_prize_count < old_prize_count:
        events['prize_taken'] = old_prize_count - new_prize_count
        events['ko'] = True

    # 3. NET Damage: damage ke lawan MINUS damage ke diri sendiri
    opp_index = 1 - player_index
    old_opp_active = old_state.players[opp_index].active
    new_opp_active = new_state.players[opp_index].active
    old_my_active = my_old.active
    new_my_active = my_new.active

    damage_dealt = 0
    if old_opp_active and old_opp_active[0] and new_opp_active and new_opp_active[0]:
        hp_loss = old_opp_active[0].hp - new_opp_active[0].hp
        if hp_loss > 0:
            damage_dealt = hp_loss

    damage_received = 0
    if old_my_active and old_my_active[0] and new_my_active and new_my_active[0]:
        hp_loss_self = old_my_active[0].hp - new_my_active[0].hp
        if hp_loss_self > 0:
            damage_received = hp_loss_self

    net_damage = damage_dealt - damage_received
    if net_damage > 0:
        events['net_damage'] = net_damage
    elif damage_received > 0:
        # Track damage received sebagai negatif (penalty)
        events['damage_received'] = damage_received

    # ──────────────────────────────────────────────
    # 4. Evolusi Active: verifikasi via serial tracking (bukan heuristic energi)
    # ──────────────────────────────────────────────
    if old_my_active and old_my_active[0] and new_my_active and new_my_active[0]:
        old_id = old_my_active[0].id
        new_id = new_my_active[0].id
        if old_id != new_id:
            old_serial = old_my_active[0].serial
            # Cek apakah old active pindah ke bench (switch/retreat)
            old_on_bench = any(p.serial == old_serial for p in my_new.bench if p)
            if not old_on_bench:
                # Bisa evolusi atau pengganti setelah KO.
                # Exclude KO: bench berkurang karena 1 pindah ke active.
                old_bench_n = sum(1 for p in my_old.bench if p)
                new_bench_n = sum(1 for p in my_new.bench if p)
                if new_bench_n >= old_bench_n:
                    events['evolved'] = True

    # ──────────────────────────────────────────────
    # 5. Evolusi Bench: cek serial bench yg hilang tanpa pengurangan bench
    # ──────────────────────────────────────────────
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
                break  # cukup 1 event per step

    # ──────────────────────────────────────────────
    # 6. Item / Supporter: deteksi dari game logs
    # ──────────────────────────────────────────────
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

    return events


def calculate_step_reward(new_state, player_index: int, events: dict = None, end_reason: int = 0) -> float:
    """
    Reward dengan anti-hacking safeguards.

    Prinsip:
    1. Terminal Reward (±3.0) >> Intermediate Reward per step (max ~0.5)
       → Model HARUS menang untuk reward maksimal
    2. Step penalty naik seiring waktu → stall makin mahal
    3. Damage = NET (dealt - received) → trade blow gak ngasih reward
    4. Diminishing returns per event type → spam dikurangi
    5. **Deck-out (end_reason=2) → terminal reward = 0**
       → Model tidak belajar apa-apa dari deck-out, abaikan.
    6. Evolusi diverifikasi via serial tracking (bukan heuristic energi)
       → Switch/retreat gak salah detek sebagai evolve
    7. Item/Supporter reward kecil + decaying → hanya bantu eksplorasi awal

    Anti-Hacking Matrix:
    ┌─────────────────────────────┬──────────────────┬──────────────────────────┐
    │ Skenario                    │ Dampak           │ Mitigasi                │
    ├─────────────────────────────┼──────────────────┼──────────────────────────┤
    │ Energy attach spam          │ Farming +0.1/ea  │ Diminishing + cek       │
    │ Damage trade loop           │ +0.1 each side   │ Net damage (dealt -     │
    │                             │                  │ received)               │
    │ Stalling                    │ Kumpulin reward  │ Step penalty naik per   │
    │                             │                  │ turn (+scaling)          │
    │ Self-damage exploit         │ Double count     │ Damage received         │
    │                             │                  │ mengurangi reward       │
    │ Symmetric self-play         │ Net zero learning│ Terminal ±3.0           │
    │                             │                  │ memecah simetri          │
    │ Deck-out win/loss           │ Stalling farm    │ Terminal = 0            │
    │                             │                  │ (diabaikan training)    │
    │ Prize delay                 │ Skip KO, farm    │ Prize reward            │
    │                             │ damage           │ (0.50) >> damage        │
    │                             │                  │ (max 0.25)              │
    │ Switch/retreat sebagai      │ False positive   │ Serial tracking: cek    │
    │ evolusi                     │ evolve reward    │ apakah old active       │
    │                             │                  │ pindah ke bench         │
    │ Item/Supporter spam         │ Farming reward   │ Diminishing returns     │
    │                             │ tanpa efek       │ (0.80**n → cepat habis) │
    │                             │ in-game          │                         │
    └─────────────────────────────┴──────────────────┴──────────────────────────┘
    """
    if new_state is None:
        return 0.0

    # 1. Step penalty naik seiring turn (stall makin mahal)
    turn = new_state.turn
    r_step = -0.02 * (1.0 + turn / 100.0)

    # 2. Intermediate event rewards dengan diminishing returns
    r_event = 0.0

    if events:
        # Energy attach: diminishing returns (-20% per attach)
        if events.get('energy_attached', 0) > 0:
            n = _increment_counter('energy')
            decay = 0.80 ** n  # 1st: 0.10, 2nd: 0.08, 3rd: 0.064, ...
            r_energy = 0.10 * events['energy_attached'] * decay
            r_event += r_energy

        # Evolution (hanya 1x per Pokemon per game yang berarti)
        if events.get('evolved'):
            # Major evolution: reward lebih besar dari energy
            n = _increment_counter('evolve')
            decay = 0.70 ** n  # 1st: 0.15, 2nd: 0.105, 3rd: 0.074, ...
            r_event += 0.15 * decay

        # Bench evolution (reward lebih kecil dari active evolution)
        if events.get('bench_evolved'):
            n = _increment_counter('bench_evolve')
            decay = 0.70 ** n
            r_event += 0.10 * decay

        # Supporter reward: dorong eksplorasi draw engine (decaying)
        if events.get('supporter_played'):
            n = _increment_counter('supporter')
            decay = 0.80 ** n  # 1st: 0.03, 2nd: 0.024, 3rd: 0.019, ...
            r_event += 0.03 * decay

        # Item reward: dorong eksplorasi search item (decaying)
        if events.get('item_played'):
            n = _increment_counter('item')
            decay = 0.80 ** n
            r_event += 0.02 * decay

        # Net damage: reward proporsional, tapi kecilan
        if events.get('net_damage', 0) > 0:
            r_damage = min(events['net_damage'] / 500.0, 0.25)
            r_event += r_damage

        # Damage received: penalty (trading blows gak nguntungin)
        if events.get('damage_received', 0) > 0:
            r_penalty = min(events['damage_received'] / 500.0, 0.25)
            r_event -= r_penalty

        # Prize: reward terbesar (dorong KO, bukan cuma damage)
        if events.get('prize_taken', 0) > 0:
            r_event += events['prize_taken'] * 0.50

        # KO: bonus kecil (sudah dapat prize, ini tambahan)
        if events.get('ko') and not events.get('prize_taken'):
            r_event += 0.20

    # Cap intermediate reward per step (cegah stacking)
    # 1.0 cukup untuk accommodate double prize (Pokemon ex)
    r_event = np.clip(r_event, -1.0, 1.0)

    # 3. Terminal reward (DOMINAN — menjamin kemenangan > farming)
    r_terminal = 0.0
    if new_state.result != -1:
        if new_state.result == player_index:
            # Hanya Prize win yang dikasih full reward.
            if end_reason == 1:
                r_terminal = 3.0    # Prize → full reward
            elif end_reason == 2:
                # Deck-out (reason=2): abaikan — terminal reward = 0 untuk pemenang.
                # Mencegah model belajar menang lewat stalling/mill (menunggu lawan kehabisan kartu).
                r_terminal = 0.0
            elif end_reason in (3, 4):
                r_terminal = 0.5    # NoActive/Effect → hollow win, reward kecil
            else:
                r_terminal = 1.5    # Fallback untuk reason lain
        elif new_state.result == 2:
            r_terminal = -0.5   # Draw
        else:
            r_terminal = -3.0   # Kalah besar (termasuk kalah karena deck-out sendiri)

    total = r_step + r_event + r_terminal
    return float(np.clip(total, -5.0, 5.0))
