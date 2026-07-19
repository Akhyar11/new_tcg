import re

with open('agent_rl/eval_output_final_2.txt', 'r') as f:
    lines = f.readlines()

result = "N/A"
p0_actions = []
p1_actions = []
p0_values = []
p1_values = []

current_player = None

for i, line in enumerate(lines):
    if "GAME OVER! Result:" in line:
        result = line.strip().split("Result: ")[-1]
    
    if "Player to act:" in line:
        current_player = int(line.strip().split("Player to act: P")[1])
        
    if "AI DECISION:" in line:
        parts = line.strip().split(" | Critic Value: ")
        decision = parts[0].replace("AI DECISION: ", "")
        val_str = parts[1] if len(parts) > 1 else "0.0"
        
        try:
            val = float(val_str)
        except:
            val = 0.0
            
        if current_player == 0:
            p0_actions.append(decision)
            p0_values.append(val)
        elif current_player == 1:
            p1_actions.append(decision)
            p1_values.append(val)

def summarize(actions, values, player):
    if not actions: return "No actions"
    stats = {}
    for a in actions:
        base_act = a.split('(')[0] if '(' in a else a
        stats[base_act] = stats.get(base_act, 0) + 1
        
    res = f"Player {player}:\n"
    res += f"  Total steps: {len(actions)}\n"
    res += f"  Avg Critic Value: {sum(values)/len(values):.2f}\n"
    res += f"  Min Critic Value: {min(values):.2f}\n"
    res += f"  Max Critic Value: {max(values):.2f}\n"
    res += "  Action Types:\n"
    for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True):
        res += f"    - {k}: {v}\n"
    return res

print(f"Game Result: {result}")
print(summarize(p0_actions, p0_values, 0))
print(summarize(p1_actions, p1_values, 1))

