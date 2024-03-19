def parse_input(input_str):
    # Create a dictionary to store the filesystem
    filesystem = {'/': {'size': 0, 'children': {}}}

    # Split the input string into lines
    lines = input_str.strip().split('\n')

    # Initialize the current directory to the root directory
    current_dir = []

    # Iterate over the lines
    for line in lines:
        # Split the line into words
        words = line.strip().split(' ')
        #print("words: ",words," Current dir: ",current_dir, " filesystem: ",filesystem)
        # check if command
        if words[0]=='$':
            # If the first word is "cd"
            if words[1] == 'cd':
                # If the second word is "/", change the current directory to the root directory
                if words[2] == '/':
                    current_dir = ['/']
                # If the second word is "..", change the current directory to the parent directory
                elif words[2] == '..':
                    current_dir = current_dir[:-1]
                # Otherwise, change the current directory to the specified subdirectory
                else:
                    print("append: ",words[2])
                    current_dir.append(words[2])

            # If the first word is "ls", add the children to the current directory
            elif words[1] == 'ls':
                print("list")
        else:
            add_to_path(current_dir, filesystem, words[1], words[0])



    # Return the filesystem
    return filesystem

def add_to_path(current_dir,file_system,file,file_size):
    print("add to path: ",current_dir,file,file_size)

    temp_dir=file_system
    for dir in current_dir:
        parrent_temp_dir=temp_dir[dir]
        if 'children' in parrent_temp_dir:
            temp_dir=parrent_temp_dir['children']
            if file_size.isnumeric():
                parrent_temp_dir["size"]=parrent_temp_dir["size"]+int(file_size)
    if file_size.isnumeric():
        temp_dir[file]={'size': int(file_size)}
    else:
        temp_dir[file] = {'size': 0,'children':{}}


def find_small_dirs(filesystem):
    small_dirs=[]
    iterate_elements(filesystem,small_dirs)
    return small_dirs

def iterate_elements(d,small_dirs):
    # iterate over the elements in the dictionary
    for key, value in d.items():
        # if the element is a directory, recursively iterate over its children
        print("key: ",key," value: ",value)

        if 'children'in value and value['size'] <=100000:
            small_dirs.append({'key':key,'value':value['size']})
        if 'children' in value:
            print(f"Directory: {key}")
            iterate_elements(value['children'],small_dirs)
        # otherwise, print the element (assumed to be a file)
        else:
            print(f"File: {key}")

with open('AOC_data_7', 'r') as f:
  input_str= f.read()


def solve(input_str):
    # Parse the input string into a filesystem
    filesystem = parse_input(input_str)
    print(filesystem)
    # Compute the sizes of all the directories
    directory_sizes = find_small_dirs(filesystem)
    sum=0
    for value in directory_sizes:
        sum+=value['value']

    print("sum: ",sum)

    current_size=filesystem['/']['size']
    required_space=30000000
    total_space=70000000
    needed_free=current_size+required_space-total_space

    print('the current size is',current_size, ' and to get room for a ',required_space,' sized file',needed_free,' should be cleared')


    # create a stack with the root dictionary
    stack = [filesystem]
    best_deletion=total_space
    # while the stack is not empty
    while stack:
        #print(stack)
        # pop the top dictionary from the stack
        current_dict = stack.pop()
        # iterate over the elements in the dictionary
        for key, value in current_dict.items():
            # if the element is a directory, add it to the stack
            if 'children' in value:

                temp_value=value['size']
                print(required_space,'<',temp_value,' <',best_deletion)
                if temp_value>needed_free and temp_value<best_deletion:
                    print('new best value: ',temp_value)
                    best_deletion=temp_value
                stack.append(value['children'])
            # otherwise, print the element (assumed to be a file)
            else:
                pass
                #print(f"File: {key}")
    print('the directory with the ideal size to delete is: ',best_deletion)



    # Initialize the sum of the sizes of the directories with total size
solve(input_str)
