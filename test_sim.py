import cg.game
import random

deck = [1] * 60
obs, start = cg.game.battle_start(deck, deck)
print(obs['current'])
print(obs['select'])
