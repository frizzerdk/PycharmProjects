import numpy as np


def read_grid_from_file(filename):
    # Open the file and read the lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Remove the newline characters from the lines
    lines = [line.strip() for line in lines]

    # Convert the lines to a 2D array of integers
    grid = [[int(ch) for ch in line] for line in lines]

    return np.array(grid)


# Test the function with an example file
filename = 'AOC_data_8'
grid = read_grid_from_file(filename)
print(grid)  # Expected output: [[3, 0, 3, 7, 3], [2, 5, 5, 1, 2], [6, 5, 3, 3, 3], [3, 3, 5, 4, 5], [3, 5, 3, 9, 0]]

# Define the grid


# Get the number of rows and columns in the grid
num_rows = len(grid)
num_cols = len(grid[0])

# Create the is_visible array
is_visible = np.array([[0 for _ in range(num_cols)] for _ in range(num_rows)])

visibility_score = np.array([[0 for _ in range(num_cols)] for _ in range(num_rows)])

# Print the arrays to verify that they have the correct size
print(
    is_visible)  # Expected output: [[False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]

print(num_cols)


def find_index(lst, compare_value):
    # Iterate through the list
    #print(("list: ",lst,"compare value: ",compare_value))
    if len(lst)==0:
        return 0
    for i, value in enumerate(lst):
        # If the value is equal to or larger than the compare value, return the index
        if value >= compare_value:
            return i+1
    # If no such value is found, return the max index
    return len(lst)


# Iterate through the rows and columns of the grid
for i in range(num_rows):
    for j in range(num_cols):

       # print('i/j: ',i+1,"/",j+1, "val",grid[i][j]," up: ",grid[:i     ,j]," down: ",grid[i+1:   ,j],"left: ",grid[i      ,:j],"right: ",grid[i      ,j+1:])
        up = (i == 0) or (max(grid[:i, j]) < grid[i][j])

        down = (i == num_rows - 1) or (max(grid[i + 1:, j]) < grid[i][j])

        left = (j == 0) or (max(grid[i, :j]) < grid[i][j])

        right = (j == num_cols - 1) or (max(grid[i, j + 1:]) < grid[i][j])

       # print(up, down, left, right)
        if up or down or left or right:
            is_visible[i][j] = 1

        # find best view
        up_trees = find_index((grid[:i, j])[::-1], grid[i][j])

        down_trees = find_index(grid[i + 1:, j], grid[i][j])

        left_trees = find_index((grid[i, :j])[::-1],  grid[i][j])

        right_trees = find_index(grid[i, j + 1:], grid[i][j])


        visibility_score[i][j]=up_trees*down_trees*left_trees*right_trees
        #print("up: ", up_trees, "down: ", down_trees, "left: ", left_trees, "right: ", right_trees," score: ",visibility_score[i][j])

final_tally = is_visible.sum()
print("final tally: ", final_tally, " is visible: \n", is_visible,"\n max visibility score: ", visibility_score.max(), "\n visibility score: \n", visibility_score)
