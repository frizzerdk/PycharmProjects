from time import sleep
st=0.2
# Print the first line
print("Line 1")
sleep(st)

# Print the next 4 lines, moving the cursor back to the beginning of the previous line after each print statement
print("Line 2")
sleep(st)
print("Line 3")
sleep(st)
print("Line 4")
sleep(st)
print("Line 5")
sleep(st)

# Move the cursor up to the beginning of the last line that was printed, and then overwrite it with a blank line
for i in range(3):
    print("\r", end="")
#sleep(10)
