# Import the pyzotero library
from pyzotero import zotero

# Create a Zotero object with your user ID and API key
zot = zotero.Zotero(11358086, 'user', "B2l2Ko9QIJT0XJ3PWIHJWJAC")

# Define the item data in JSON format
item_data = {
    "itemType": "book",
    "title": "Test Book",
    "creators": [
        {
            "creatorType": "author",
            "firstName": "John",
            "lastName": "Doe"
        }
    ],
    "date": "2020",
    "publisher": "Test Publisher",
    "place": "Test City",
    "ISBN": "978-0-1234-5678-9"
}

# Use the create_items method to add the item
response = zot.create_items([item_data])

# Check if the response is a dictionary and has a status key
if isinstance(response, dict) and 'status' in response:
    # Print the response status and content
    print(response['status'])
    print(response['successful'])
else:
    # Print an error message
    print(response)
