import collections
import cg.api as api

attacks = api.all_attack()
max_per_element = collections.defaultdict(int)
max_total = 0

for atk in attacks:
    total = len(atk.energies)
    if total > max_total:
        max_total = total
    counts = collections.Counter(atk.energies)
    for e_type, count in counts.items():
        if count > max_per_element[e_type]:
            max_per_element[e_type] = count

print(f"Max total energy for an attack: {max_total}")
print("Max required per specific element:")
for e, c in sorted(max_per_element.items()):
    print(f" - {api.EnergyType(e).name}: {c}")

cards = api.all_card_data()
max_retreat = max(c.retreatCost for c in cards)
print(f"Max retreat cost: {max_retreat}")
