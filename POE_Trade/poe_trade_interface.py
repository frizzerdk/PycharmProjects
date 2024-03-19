# import requests as req
import cloudscraper
from bs4 import BeautifulSoup
import json as json
from JSON_object_templates import *
import tkinter as tk
import lib

import requests

OPT_LEAGUE = 0

URL_LEAGUES = "https://www.pathofexile.com/api/trade/data/leagues"
URL_STATIC = "https://www.pathofexile.com/api/trade/data/static"
URL_STATS = "https://www.pathofexile.com/api/trade/data/stats"
URL_ITEMS = "https://www.pathofexile.com/api/trade/data/items"
URL_SEARCH = "https://www.pathofexile.com/api/trade/search/"

URL_SEARCH_FETCH_1 = "https://www.pathofexile.com/api/trade/fetch/"
URL_SEARCH_FETCH_2 = "?query="

stat_labels = {"Pseudo", "Explicit", "Implicit", "Fractured", "Enchant", "Scourge", "Crafted", "Veiled", "Monster",
               "Delve", "Ultimatum"}

class PoeTradeInterface:
    leagues = static = stats = items = json.loads("{}")

    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})

    headers = {
        'authority': 'www.pathofexile.com',
        'cache-control': 'max-age=0',
        'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'accept-language': 'en-DK,en-XA;q=0.9,en-GB;q=0.8,en-US;q=0.7,en;q=0.6,da;q=0.5',
        'cookie': 'cf_clearance=qAiBs6JJHTsyquY0XMFCoOkPLbnQJOrrU_g87ZmuBFM-1643669418-0-150; POESESSID=286e8baacf80036170fb83bc7ed07aef',
    }





    def __init__(self):
        self.load_data()
        self.scraper.headers = self.headers

    def load_data(self):

        self.leagues = self.json_form_url(URL_LEAGUES)
        self.static = self.json_form_url(URL_STATIC)
        self.stats = self.json_form_url(URL_STATS)
        self.items = self.json_form_url(URL_ITEMS)
        self.league_name = self.leagues["result"][OPT_LEAGUE]["text"]
        print(self.leagues)


    #### Functions ####
    def search_val(self, obj):
        keyVal = input("Enter a key value: \n")

        # load the json data

        # Search the key value using 'in' operator
        if keyVal in obj:
            # Print the success message and the value of the key
            print("%s is found in JSON data" % keyVal)
            print("The value of", keyVal, "is", obj[keyVal])
        else:
            # Print the message if the value does not exist
            print("%s is not found in JSON data" % keyVal)


    def filter_json(self,obj):
        dprint = True
        output_dict = [x for x in obj if x['id'] == 'Standard']
        pif(output_dict)
        result = filter(lambda x: pif(x), obj)
        pif(result, dprint)
        return


    def search(self,search_obj):
        dprint = True
        pif("==Search", dprint)
        pif(URL_SEARCH + self.league_name, dprint)
        pif(search_obj, dprint)
        r = self.scraper.post(URL_SEARCH + self.league_name, json=search_obj)
        jr = r.json()
        pj(jr, dprint)
        id = jr["id"]
        pif(id, dprint)
        results = jr["result"]
        pj(results, dprint)
        results_string = ""
        count = 0
        for i in results:
            if (count < 10):
                if results_string != "":
                    results_string += ","
                results_string += i
                count += 1
        pif(results_string, dprint)
        r2 = self.scraper.get(URL_SEARCH_FETCH_1 + results_string + URL_SEARCH_FETCH_2 + id, json=search_obj)
        jr2 = r2.json()
        pj(jr, dprint)

        return jr2

    def read_url(self,url):
        html = self.scraper.get(url).content
        return BeautifulSoup(html, 'html.parser')


    def json_form_url(self,url):
        html = self.scraper.get(url).content
        return json.loads(BeautifulSoup(html, 'html.parser').string)


def pif(mstring, show=True):
    if show:
        print(mstring)


def pj(mjson, show=True):
    if show:
        print(json.dumps(mjson, indent=2))


