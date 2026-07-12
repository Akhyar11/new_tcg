import threading
import time
from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation
import random

def worker(i):
    deck = [random.randint(1, 1267) for _ in range(60)]
    try:
        obs_dict, _ = battle_start(deck, deck)
        for _ in range(10):
            try:
                battle_select([0])
            except:
                pass
        battle_finish()
        print(f"Worker {i} success")
    except Exception as e:
        print(f"Worker {i} error: {e}")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print("Success")
