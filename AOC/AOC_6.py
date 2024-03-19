def find_marker(data, marker_length):
  # Initialize a list of the last marker_length characters received to be empty
  last_chars = []

  # Loop through each character in the data
  for i, c in enumerate(data):
    # Add the current character to the list of the last marker_length characters
    last_chars.append(c)

    # If the length of the list of the last marker_length characters is greater than marker_length, remove the first character in the list
    if len(last_chars) > marker_length:
      last_chars.pop(0)

    # If all of the characters in the list of the last marker_length characters are different, return the number of characters processed so far
    if len(set(last_chars)) == marker_length:
      return i + 1

  # If no marker is found, return -1
  return -1

# Read the data from the AOC_data_6 file
with open('AOC_data_6', 'r') as f:
  data = f.read()

# Test the find_marker function for start-of-packet markers (marker_length = 4)
print(find_marker(data, 4))

# Test the find_marker function for start-of-message markers (marker_length = 14)
print(find_marker(data, 14))
