import torch
import torchvision
from torch.utils.data import DataLoader


# Load the MNIST dataset
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Print the size of the dataset
print(len(mnist_train), len(mnist_test))

# Access a specific example
print(mnist_train[0])

train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=True)
