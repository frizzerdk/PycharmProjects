import random
import sys
import os
startGold = 100000000
startBank = 100000
bank = startBank
gold = startGold
startBet = 1
bet = startBet
tries = 0
plays = 0
while gold > 0:
    print("%s %s %s %s %s %s %s" % (tries, "gold", gold, "bet", bet, "gain", gold / startGold))
    tries += 1
    if random.randrange(0, 2) == 1:
        gold += bet
        bet = startBet
    else:
        gold -= bet
        bet *= 3




