# Define the input list of lists
lst = [['G', 'J', 'Z'], ['C', 'V', 'F', 'W', 'P', 'R', 'L', 'Q'], ['R', 'G', 'L', 'C', 'M', 'P', 'F'], ['M', 'H', 'P', 'W', 'B', 'F', 'L'], ['Q', 'V', 'S', 'F', 'C', 'G'], ['L', 'T', 'Q', 'M', 'Z', 'J', 'H', 'W'], ['V', 'B', 'S', 'F', 'H'], ['S', 'Z', 'J', 'F'], ['T', 'B', 'H', 'F', 'P', 'D', 'C', 'M']]

def print_stacks(stacks):
    # The number of rows in the output is the length of the longest sublist in the stacks list
    num_rows = max([len(sublist) for sublist in stacks])

    # The number of columns in the output is the length of the stacks list
    num_cols = len(stacks)

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
        print(row_str)

    # Add column numbering
    col_nums = [f"{col+1:2d}" for col in range(num_cols)]
    print("" + "  ".join(col_nums))
print_stacks(lst)
