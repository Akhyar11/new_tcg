import numpy as np

# Counter global per-game untuk diminishing returns (direset via reset_trackers())
_event_counters = {}


def reset_trackers():
    """Reset event counters untuk game baru (dipanggil di auto-reset)."""
    _event_counters.clear()


def _increment_counter(key: str) -> int:
    """Increment counter dan return nilai sebelum increment."""
    val = _event_counters.get(key, 0)
    _event_counters[key] = val + 1
    return val


def detect_events(old_state, new_state, player_index: int) -> dict:
    """
    Mendeteksi event apa yang terjadi dalam satu step dengan membandingkan
    state sebelum dan sesudah aksi dieksekusi.

    Anti-hacking:
    - Hanya mendeteksi perubahan NET (bukan gross)
    - Damage dicek dari state aktual (bisa diverifikasi)
    - Prize hanya terdeteksi jika prize beneran berkurang

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

    # 4. Evolusi: cek ID berubah (hanya naik, bukan ganti Pokemon)
    if old_my_active and old_my_active[0] and new_my_active and new_my_active[0]:
        old_id = old_my_active[0].id
        new_id = new_my_active[0].id
        if old_id != new_id:
            # Verifikasi: ini evolusi (bukan retreat/switch) dengan cek bench
            # Kalau jumlah energi di bench naik, berarti switch, bukan evolve
            old_bench_count = sum(len(p.energies) for p in my_old.bench if p)
            new_bench_count = sum(len(p.energies) for p in my_new.bench if p)
            if new_bench_count == old_bench_count:
                events['evolved'] = True
            # Jika bench energi berubah, kemungkinan switch — jangan reward sebagai evolve

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

    Anti-Hacking Matrix:
    ┌──────────────────────┬──────────────────┬───────────────────────┐
    │ Skenario             │ Dampak           │ Mitigasi             │
    ├──────────────────────┼──────────────────┼───────────────────────┤
    │ Energy attach spam   │ Farming +0.1/ea  │ Diminishing + cek    │
    │ Damage trade loop    │ +0.1 each side   │ Net damage (dealt -   │
    │                      │                  │ received)             │
    │ Stalling             │ Kumpulin reward  │ Step penalty naik per │
    │                      │                  │ turn (+scaling)       │
    │ Self-damage exploit  │ Double count     │ Damage received       │
    │                      │                  │ mengurangi reward     │
    │ Symmetric self-play  │ Net zero learning│ Terminal ±3.0         │
    │                      │                  │ memecah simetri       │
    │ Deck-out win/loss    │ Stalling farm    │ Terminal = 0          │
    │                      │                  │ (diabaikan training)  │
    │ Prize delay          │ Skip KO, farm    │ Prize reward          │
    │                      │ damage           │ (0.50) >> damage      │
    │                      │                  │ (max 0.25)            │
    └──────────────────────┴──────────────────┴───────────────────────┘
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
        # Deck-out (reason=2): abaikan — terminal reward = 0
        # Model TIDAK belajar apa-apa dari game yang berakhir deck-out.
        if end_reason == 2:
            r_terminal = 0.0
        elif new_state.result == player_index:
            r_terminal = 3.0    # Menang besar
        elif new_state.result == 2:
            r_terminal = -0.5   # Draw
        else:
            r_terminal = -3.0   # Kalah besar

    total = r_step + r_event + r_terminal
    return float(np.clip(total, -5.0, 5.0))
