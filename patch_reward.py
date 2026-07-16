with open("agent_rl/reward.py", "r") as f:
    content = f.read()

# Fix 1: _increment_counter definition
old_inc = """def _increment_counter(key: str) -> int:
    \"\"\"Increment counter dan return nilai sebelum increment.\"\"\"
    val = _event_counters.get(key, 0)
    _event_counters[key] = val + 1
    return val"""
new_inc = """def _increment_counter(key: str, player_index: int) -> int:
    \"\"\"Increment counter dan return nilai sebelum increment.\"\"\"
    full_key = f"{key}_{player_index}"
    val = _event_counters.get(full_key, 0)
    _event_counters[full_key] = val + 1
    return val"""
content = content.replace(old_inc, new_inc)

# Fix 2: ready_serials in reset_trackers
old_reset = """def reset_trackers():
    \"\"\"Reset event counters untuk game baru.\"\"\"
    _event_counters.clear()
    _event_counters['ready_serials'] = set()"""
new_reset = """def reset_trackers():
    \"\"\"Reset event counters untuk game baru.\"\"\"
    _event_counters.clear()
    _event_counters['ready_serials_0'] = set()
    _event_counters['ready_serials_1'] = set()"""
content = content.replace(old_reset, new_reset)

# Fix 3: ready_serials in detect_events
old_ready = """    # Ensure ready_serials exists
    if 'ready_serials' not in _event_counters:
        _event_counters['ready_serials'] = set()
    ready_serials = _event_counters['ready_serials']"""
new_ready = """    # Ensure ready_serials exists
    key_ready = f'ready_serials_{player_index}'
    if key_ready not in _event_counters:
        _event_counters[key_ready] = set()
    ready_serials = _event_counters[key_ready]"""
content = content.replace(old_ready, new_ready)

# Fix 4: _increment_counter calls in calculate_step_reward
calls_to_replace = [
    ("n = _increment_counter('bench')", "n = _increment_counter('bench', player_index)"),
    ("n = _increment_counter('energy')", "n = _increment_counter('energy', player_index)"),
    ("n = _increment_counter('evolve')", "n = _increment_counter('evolve', player_index)"),
    ("n = _increment_counter('bench_evolve')", "n = _increment_counter('bench_evolve', player_index)"),
    ("n = _increment_counter('supporter')", "n = _increment_counter('supporter', player_index)"),
    ("n = _increment_counter('item')", "n = _increment_counter('item', player_index)"),
    ("n = _increment_counter('battle_ready')", "n = _increment_counter('battle_ready', player_index)"),
    ("n = _increment_counter('retreat')", "n = _increment_counter('retreat', player_index)")
]
for old_call, new_call in calls_to_replace:
    content = content.replace(old_call, new_call)

# Fix 5: deck-out symmetric reward
old_term = """        if draw:
            r_terminal = 0.0
        else:
            # Menang = +2.0, Kalah = -2.0 untuk semua alasan kemenangan yang valid
            r_terminal = 2.0 if won else -2.0"""
new_term = """        if draw or end_reason == 2:
            r_terminal = 0.0
        else:
            # Menang = +2.0, Kalah = -2.0 untuk semua alasan kemenangan yang valid
            r_terminal = 2.0 if won else -2.0"""
content = content.replace(old_term, new_term)

with open("agent_rl/reward.py", "w") as f:
    f.write(content)
