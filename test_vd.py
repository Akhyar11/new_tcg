import cg.game as g, pandas as pd, json
d = pd.read_csv('agent_rl/deck/gen_deck_000.csv', header=None)
g.battle_start(d[0].tolist(), d[0].tolist())
g.battle_select([0]) # select coin
vd = json.loads(g.visualize_data())[-1]
print(type(vd['current']['players'][1]['hand'][0]))
