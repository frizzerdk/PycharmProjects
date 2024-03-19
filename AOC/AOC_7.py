# Parse input and build data structure
tree = {}
current = tree
for line in open("AOC_data_7"):
    line = line.strip()
    if line.startswith("$"):
        parts = line.split(" ")
        if parts[1] == "cd":
            if parts[2] == "/":
                current = tree
            elif parts[2] == "..":
                current = current[".."]
            else:
                current = current[parts[2]]
    elif line.startswith("dir"):
        dirname = line.split(" ")[1]
        current[dirname] = {"..": current}
    else:
        parts = line.split(" ")
        filename, size = parts[0], parts[1]
        current[filename] = size

# Calculate total size of directories using a stack
def get_total_size(node):
    total = 0
    stack = [node]
    processed = []
    while stack:
        print("stack: ",stack)
        current = stack.pop()
        if isinstance(current, dict):
            for value in current.values():
                if isinstance(value, str) and value.isdigit():
                    total += int(value)
                elif isinstance(value, dict) and value not in processed:
                    stack.append(value)
                    processed.append(value)
    return total
# Find all directories with total size <= 100000
print(tree)
if False:
    small_dirs = []
    for name, node in tree.items():
        if isinstance(node, dict):
            total_size = get_total_size(node)
            if total_size <= 100000:
                small_dirs.append((name, total_size))

    # Print answer
    print(sum(total_size for name, total_size in small_dirs))
