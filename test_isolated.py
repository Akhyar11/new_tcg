import os
import subprocess

deck_dir = "agent_rl/deck"
files = sorted([f for f in os.listdir(deck_dir) if f.endswith(".csv")])

print(f"Testing {len(files)} decks in separate processes...")

test_script = """
import sys
from cg.game import battle_start

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
    return deck

try:
    deck = load_deck(sys.argv[1])
    obs, _ = battle_start(deck, deck)
    if obs is None or 'current' not in obs or obs['current'] is None:
        sys.exit(1)
    sys.exit(0)
except Exception as e:
    sys.exit(1)
"""

with open("test_one.py", "w") as f:
    f.write(test_script)

failures = []
for file in files:
    filepath = os.path.join(deck_dir, file)
    res = subprocess.run(["venv/bin/python", "test_one.py", filepath], capture_output=True)
    if res.returncode != 0:
        failures.append(file)

print(f"Total isolated failures: {len(failures)} out of {len(files)}")
if len(failures) > 0:
    print(f"First 10 failures: {failures[:10]}")
