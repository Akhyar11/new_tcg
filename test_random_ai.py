import cg.game
import random

deck = []
with open("agent_rl/deck/gen_deck_000.csv", "r") as f:
    deck = [int(line.strip()) for line in f if line.strip().isdigit()]

obs, start_data = cg.game.battle_start(deck, deck)

turn = 0
actions = {"ATTACK": 0, "PLAY": 0, "ATTACH": 0, "EVOLVE": 0, "END": 0}

while obs and not obs.get("current", {}).get("isGameOver", False):
    turn += 1
    select_data = obs.get("select")
    if not select_data or not select_data.get("option"):
        break
    
    opts = select_data["option"]
    opt_count = len(opts)
    
    idx = random.randint(0, opt_count-1)
    chosen_opt = opts[idx]
    
    # We can get the string name roughly by just getting the type index? No, we don't have OptionType easily imported without cg.api
    # But from cg.api: 7=PLAY, 8=ATTACH, 9=EVOLVE, 13=ATTACK, 14=END
    otype = chosen_opt.get("type", -1)
    if otype == 13:
        actions["ATTACK"] += 1
    elif otype == 7:
        actions["PLAY"] += 1
    elif otype == 8:
        actions["ATTACH"] += 1
    elif otype == 9:
        actions["EVOLVE"] += 1
    elif otype == 14:
        actions["END"] += 1

    obs = cg.game.battle_select([idx])
    
    if turn > 1000:
        print("Timeout 1000 actions")
        break

print(f"Game over. Total actions: {turn}")
print(f"Action counts: {actions}")
