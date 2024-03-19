# import requests as req

from poe_trade_interface import *
from ui_trade import *

import tkinter as tk
import lib

#### MAIN CODE ######
def main():
    pt= PoeTradeInterface()
    ui =  ItemSearchUi(pt)
    print(pt.leagues)

    print(pt.league_name)
    pt.search_val(pt.stats)
    pif(pt.leagues)







main()
