from time import sleep
input_string = """    [C]             [L]         [T]
    [V] [R] [M]     [T]         [B]
    [F] [G] [H] [Q] [Q]         [H]
    [W] [L] [P] [V] [M] [V]     [F]
    [P] [C] [W] [S] [Z] [B] [S] [P]
[G] [R] [M] [B] [F] [J] [S] [Z] [D]
[J] [L] [P] [F] [C] [H] [F] [J] [C]
[Z] [Q] [F] [L] [G] [W] [H] [F] [M]
 1   2   3   4   5   6   7   8   9 """
instruction_string = """move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2"""
with open("AOC_data_5", "r") as f:
    text = f.read().strip()
instruction_string=text
def main():


    # Split the input string into a list of strings, one for each row
    rows = input_string.split("\n")

    # Create a list of stacks
    stacks = [[] for _ in range(9)]

    # Iterate over each character in the input string
    for row in rows:
        i = 0
        consecutive_spaces = 0
        for char in row:
            # If the character is a space, increment the number of consecutive spaces
            if char == " ":
                consecutive_spaces += 1
            else:
                consecutive_spaces = 0

            # If the number of consecutive spaces is four, increment the index and reset the space counter
            if consecutive_spaces == 4:
                i += 1
                consecutive_spaces = 0

            # If the character is a letter, append it to the appropriate stack and increment the space counter
            if char.isalpha():
                stacks[i].insert(0, char)
                i += 1
               # consecutive_spaces += 1
    # Print the stacks
    for i, stack in enumerate(stacks):
        print(f"Stack {i + 1}: {stack}")

    print_stacks(stacks)
    stacks2=move_entries(stacks,instruction_string)
    print_stacks(stacks)
    # Create an empty string to store the last characters
    last_chars = ''

    # Loop through the subarrays and extract the last character from each one
    for subarr in stacks:
        last_chars += subarr[-1]
    print(last_chars)
    #
#    with open("AOC_data_5", "r") as f:
#        # Read the contents of the file into a string
 #       text = f.read().strip()
pstring = ""

def print_stacks(stacks):
    # Use the global keyword to specify that you want to use the global pstring variable
    global pstring

    # Reset the value of the pstring variable
    pstring = ""

    # Define the mprint function
    def mprint(istring):
        global pstring
        # Append to the global pstring variable
        pstring = pstring + "\n" + istring

        # Return without a value
        return

    # The number of rows in the output is the length of the longest sublist in the stacks list
    num_rows = max([len(sublist) for sublist in stacks])

    # The number of columns in the output is the length of the stacks list
    num_cols = len(stacks)

    #mprint("{}".format(stacks))

    # Add left and right padding to each string in the grid so that they have equal length
    padded_stacks = [[f"[{item}] " for item in stack] for stack in stacks]

    # Iterate over the rows in the output
    for row in reversed(range(num_rows)):
        # Initialize an empty string to hold the row
        row_str = ""



        # Iterate over the columns in the output
        for col in range(num_cols):
            # If the row and column correspond to an element in the input list, add the element to the row
            if row < len(padded_stacks[col]) and col < len(padded_stacks):
                row_str += padded_stacks[col][row]
            else:
                row_str +="    "

        # Print the row
        mprint(row_str)

    # Add column numbering
    col_nums = [f"{col+1:2d}" for col in range(num_cols)]
    mprint("" + "  ".join(col_nums))
    print(pstring)
    sleep(0.1)

def move_entries_single(arr, instruction_string):
  # Split the instruction string into a list of individual instructions
  instructions = instruction_string.split('\n')

  # Loop through the instructions and apply each one
  for instruction in instructions:
    # Split the instruction into its component parts
    parts = instruction.split()
    # Extract the number of entries to move, the source array, and the destination array
    num_entries = int(parts[1])
    src_idx = int(parts[3]) - 1
    dst_idx = int(parts[5]) - 1

    # Loop through the number of entries to move
    for i in range(num_entries):
      # Extract the entry to move from the source array
      entry_to_move = arr[src_idx][-1]

      # Remove the entry from the source array
      arr[src_idx] = arr[src_idx][:-1]

      # Add the entry to the destination array
      arr[dst_idx] += [entry_to_move]

  # Return the resulting nested array
    print_stacks(arr)
  return arr
def move_entries(arr, instruction_string):

  # Split the instruction string into a list of individual instructions
  instructions = instruction_string.split('\n')

  # Loop through the instructions and apply each one
  for instruction in instructions:
    # Split the instruction into its component parts
    parts = instruction.split()
    # Extract the number of entries to move, the source array, and the destination array
    num_entries = int(parts[1])
    src_idx = int(parts[3]) - 1
    dst_idx = int(parts[5]) - 1

    # Extract the entries to move from the source array
    entries_to_move = arr[src_idx][-num_entries:]

    # Remove the entries from the source array
    arr[src_idx] = arr[src_idx][:-num_entries]

    # Add the entries to the destination array
    arr[dst_idx] += entries_to_move
    print_stacks(arr)
  # Return the resulting nested array
  return arr
main()

