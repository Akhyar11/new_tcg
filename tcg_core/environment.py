from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_dataclass, Observation

class TCGEnvironment:
    """
    A Gym-like wrapper for the C++ TCG Engine.
    """
    def __init__(self):
        self.obs = None
        self.done = True
        self.deck0 = None
        self.deck1 = None

    def reset(self, deck0, deck1):
        """
        Starts a new game with the provided decks.
        Returns the initial observation and done flag.
        """
        self.deck0 = deck0
        self.deck1 = deck1
        obs_dict, _ = battle_start(self.deck0, self.deck1)
        self.obs = to_dataclass(obs_dict, Observation)
        self.done = False
        return self.obs, self.done

    def step(self, choices):
        """
        Executes actions in the game.
        Returns (obs, reward, done, info).
        """
        if self.done:
            raise Exception("Cannot step in a finished environment. Call reset() first.")
            
        try:
            obs_dict = battle_select(choices)
            self.obs = to_dataclass(obs_dict, Observation)
        except Exception:
            # Failsafe for invalid/empty actions
            try:
                opt_count = len(self.obs.select.option) if self.obs.select and self.obs.select.option else 0
                min_c = self.obs.select.minCount if self.obs.select else 0
                obs_dict = battle_select(list(range(min(opt_count, min_c))))
                self.obs = to_dataclass(obs_dict, Observation)
            except Exception as e:
                self.done = True
                return self.obs, 0, self.done, {"error": str(e)}

        self.done = (self.obs.current is None) or (self.obs.current.result != -1)
        
        info = {
            "turn": self.obs.current.turn if self.obs.current else 0,
            "result": self.obs.current.result if self.obs.current else -1
        }
        
        return self.obs, 0, self.done, info

    def close(self):
        """
        Finishes the battle and cleans up the C++ engine state.
        """
        battle_finish()
        self.done = True
