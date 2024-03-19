from json import *

trade_query_default= {
    "query": {
        "status": {
            "option": "online"
        },
        "name": "The Pariah",
        "stats": [{
            "type": "sum",
            "filters": [{
                "id": "pseudo.pseudo_total_attack_speed",
                "value": {
                    "min": 1,
                    "max": 9999
                },
                "disabled": False
            }, {
                "id": "pseudo.pseudo_count_elemental_resistances",
                "value": {
                    "min": 1,
                    "max": 9999
                },
                "disabled": False
            }]
        }],
    },
    "sort": {
        "statgroup.0": "dsc"
    },
    "filters": {
        "type_filters": {
            "filters": {
                "category": {
                    "option": "ring"
                },
                "rarity": {
                    "option": "unique"
                }
            }
        }
    }

}
