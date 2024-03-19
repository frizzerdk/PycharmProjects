import numpy as np

Q = 13

size = 100
string = 'int input['+str(size)+']={ '
string += '0'
for i in range(1, size):
    string += " , "
    string += str(int(np.sin((i / size) * 2 * np.pi) * 2 ** Q))
string += ' };'
print(string)

for f in range(0, 6):
    size = 73
    string = 'int filter' + str(f) + '['+str(size)+']={ '


    string += str(f)
    for i in range(1, size):
        string += " , "
        string += str(int(((i / size)) * 2 ** Q))
    string += ' };'
    print(string)

size = 256

string = 'int crossfade_out['+str(size)+']={ '

i = 0
string += str(int(np.cos((i / (size-1)) * np.pi/2) * 2 ** Q))
for i in range(1, size):
    string += " , "
    string += str(int(np.cos((i / (size-1)) * np.pi/2) * 2 ** Q))
string += ' };'
print(string)

size = 256
string = 'int crossfade_in['+str(size)+']={ '
i = 0
string += str(int(np.sin((i / (size-1)) * np.pi/2) * 2 ** Q))
for i in range(1, size):
    string += " , "
    string += str(int(np.sin((i / (size-1)) * np.pi/2) * 2 ** Q))
string += ' };'
print(string)

print(2**13)