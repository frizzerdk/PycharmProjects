import requests
import json
import pyperclip

def submit_cards(cards, deck_name=None):
    if deck_name is None:
        # Define the parameters for getting the deck names
        params = {
            "action": "deckNames",
            "version": 6
        }

        # Send the request to Anki
        response = requests.post("http://localhost:8765", json=params)

        # Get the deck names
        deck_names = response.json()["result"]

        # Print the deck names with numbers
        for i, name in enumerate(deck_names):
            print(f"{i+1}. {name}")

        # Ask the user to choose a deck
        choice = int(input("Enter the number of the deck you want to add cards to: "))

        # Get the chosen deck name
        deck_name = deck_names[choice-1]

        # Print the chosen deck name
        print(f"You chose {deck_name}")

    # Loop through the cards and add them to the chosen deck
    for card in cards:
        # Define the parameters for adding a card
        params = {
            "action": "addNote",
            "version": 6,
            "params": {
                "note": {
                    "deckName": deck_name, # Use the chosen deck name here
                    "modelName": "Basic",
                    "fields": card, # Use the card fields here
                    "options": {
                        "allowDuplicate": False
                    },
                    "tags": [
                        "test"
                    ]
                }
            }
        }

        # Send the request to Anki
        response = requests.post("http://localhost:8765", json=params)

        # Print the response
        print(response.json())

#'[{"Front": "Hello", "Back": "World"}, {"Front": "Apple", "Back": "りんご"}, {"Front": "2 + 2", "Back": "4"}]'


def string_to_cards(input_string):
    cards = json.loads(input_string)
    return cards

def prompt_for_cards():
    instruction_string = 'Help me format points above for flashcards i can add to my anki deck with as much detail as possible.' \
                         ' You can use simple latex formatting if it makes sense but instead of enclosing the equation in :' \
                         ' $ $ , it needs to be enclosed in : \( \) .' \
                         'In order to use LaTeX in JSON strings in Python, you need to use double backslashes \\\\ in your LaTeX commands.' \
                         ' They should be in a json string as follows:' \
                         '\n[{"Front": "Hello", "Back": "World"}, {"Front": "Apple", "Back": "りんご"}, {"Front": "2 + 2", "Back": "4"}]'
    pyperclip.copy(instruction_string)  # Copy the instruction string to the clipboard
    print("Instructions copied to clipboard!")

    print("Enter your cards in JSON format (end with an empty line):")
    input_lines = []
    while True:
        line = input()
        if line:
            input_lines.append(line)
        else:
            break
    input_string = '\n'.join(input_lines)

    cards = string_to_cards(input_string)
    return cards


cards = prompt_for_cards()
submit_cards(cards)
