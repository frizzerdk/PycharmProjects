import random
import sys
import os
startGold = 10000
startBank = 100000
bank = startBank
gold = startGold
startBet = 1
bet = startBet
tries = 0
n_plays = 0
# arrays to save the data
rounds = []
plays = []
while bank > 0:
    print("%s %s %s %s %s " % (n_plays, "bank", bank, "gain", bank / startBank))
    n_plays += 1
    startGold = bank/10
    gold = startGold
    bet = startBet
    tries = 0
    bank -= gold
    plays.append(bank)
    
    while gold > 0:
        print("%s %s %s %s %s %s %s" % (tries, "gold", gold, "bet", bet, "gain", gold / startGold))
        tries += 1
        roll = random.randrange(0, 2)
        print(roll)
        if roll== 1:
            gold += bet
            bet = startBet
        else:
            gold -= bet
            if bet > startGold/10:
                bet *= 10
            else:
                bet *= 10
            if gold - bet < startGold/2:
                bank += gold

                gold = -1
    







