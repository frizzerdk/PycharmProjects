import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# My imports
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as Funct

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import Image
import enum
import json
def main():
    device = load_device()







    # preprocess_data(X_train_tensor,Y_train_tensor)

    data_set_train, data_set_val, data_set_test = download_mnist()

    dataloader_train, dataloader_val = make_dataloaders(data_set_train, data_set_val)

    input_size, output_size = get_model_IO_size(data_set_train)

    model = MyNetwork(input_size, output_size)

    loss_fcn, optimizer = get_loss_optimizer_fcns(model)

    num_epochs=30
    print(data_set_train.tensors[0].shape)
    print(dataloader_val.dataset.tensors[0].shape)
    print('input_size, output_size',input_size, output_size)


    train_losses, val_losses,train_losses_epoch, val_losses_epoch, val_accuracy_epoch, epochs= train(num_epochs,model,dataloader_train,optimizer,loss_fcn,dataloader_val)
    print(epochs)
    plot_loss(train_losses,val_losses)
    plot_loss(train_losses_epoch,val_losses_epoch)
    mplot(val_accuracy_epoch,"Accuracy ")

def download_mnist():
    # Load the MNIST dataset
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    #print(mnist_train.data.dtype)


    train_data_tensor= mnist_train.data.type(torch.float32)
    train_target_tensor= mnist_train.targets.type(torch.float32)
    test_data_tensor = mnist_test.data.type(torch.float32)
    test_target_tensor = mnist_test.targets.type(torch.float32)

    #print(train_data_tensor.dtype)

    validation_split=[0.2,0.8]

    X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor = split_for_validation(train_data_tensor,train_target_tensor, validation_split)

    # preprocess_data(X_train_tensor,Y_train_tensor)

    data_set_train, data_set_val = make_datasets(X_train_tensor.reshape(X_train_tensor.shape[0],-1), Y_train_tensor, X_val_tensor.reshape(X_val_tensor.shape[0],-1), Y_val_tensor)

    data_set_test = TensorDataset(test_data_tensor.reshape(test_data_tensor.shape[0],-1),test_target_tensor)

    return data_set_train,data_set_val,data_set_test



def load_device():
    return 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_to_tensor(train_data,test_data):
    return torch.tensor(train_data.values.astype('float32')),torch.tensor(test_data.values.astype('float32'))


def split_data(tensor, fractions, dim=0):
    if sum(fractions) < 1:
        fractions.append(1 - sum(fractions))
    sizes = [int(float(tensor.shape[dim]) * fraction) for fraction in fractions]
    if sum(sizes) < tensor.shape[dim]:
        sizes[-1] = sizes[-1] + tensor.shape[dim] - sum(sizes)

    return torch.split(tensor, sizes, dim=dim)


def split_for_validation(data, target, fractions):
    data_array = split_data(data, fractions)
    target_array = split_data(target,fractions)
    return data_array[0], target_array[0], data_array[1], target_array[1]

def preprocess_data(X,y):
    X_2d = X.reshape(X.shape[0], 28, 28)
    y_2d = y
    #X_2d_test = X_test.values.reshape(X_test.shape[0], 28, 28).astype('float32')
    do_plot=True
    if do_plot:
        # Create a figure and a set of subplots
        fig, ax_array = plt.subplots(nrows=10,ncols=10, figsize=(20,40),sharey=True, sharex=True)

        # Define a function that updates the values in the subplots at each frame
        # This function takes the current frame number as an argument, and uses it
        # to index into the array and update the values in the subplots
        def update(u):

            idx = 0
            for i in range(10):
                for j in ((X_2d[y_2d==i])[u*10:u*10+10]):
                    print("i/10: ",i," u/10: ",u,end='\r')
                    digit = i
                    idx += 1
                    plt.subplot(10,10,idx)
                    plt.imshow(j, cmap='cividis')
                    plt.title(f'Digit is {digit}',fontsize=14)
                    plt.grid(None)

        # Create an animation using the FuncAnimation class
        # This animation will call the update() function at regular intervals
        # to update the values in the subplots
        anim = animation.FuncAnimation(fig, update, frames=10, interval=100)

        # Save the animation as a GIF file using ImageMagick as the writer
        anim.save('animation.gif', writer='imagemagick')

def make_datasets(X_tensor, y_tensor,X_valid_tensor, y_valid_tensor):
    print(X_tensor.size(),y_tensor.size())
    data_set_train = TensorDataset(X_tensor, y_tensor)
    data_set_val = TensorDataset(X_valid_tensor, y_valid_tensor)
    return data_set_train,data_set_val

# Function to save the model
def saveModel(model,path):
    path = "/kaggle/working/NetModel.pth"
    torch.save(model.state_dict(), path)

def make_dataloaders(training_set,validation_set):
    train_loader = DataLoader(training_set, batch_size=100, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=100, shuffle=False)
    return train_loader,val_loader

def get_model_IO_size(dataset):
    return dataset[0][0].shape[0],dataset[:][1].unique().shape[0]

def get_loss_optimizer_fcns(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    return loss_fn, optimizer


# Test Network
class MyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyNetwork, self).__init__()

        self.layer1 = nn.Linear(input_size, 28)  # Pytorch Linear Layer = Keras Dense
        self.layer2 = nn.Linear(28, 28)
        self.layer3 = nn.Linear(28, 28)
        self.layer4 = nn.Linear(28, output_size)

    def forward(self, x):
        x1 = Funct.relu(self.layer1(x))
        x2 = Funct.relu(self.layer2(x1))
        x3 = Funct.relu(self.layer3(x2))
        x4 = self.layer4(x3)
        return x4

    def evaluate(self, eval_loader, loss_fn):
        val_loss = 0
        # Set the model to evaluation mode
        self.eval()
        total_correct = 0
        with torch.inference_mode():
            for data in eval_loader:
                input_data, labels = data

                # _, prediction= torch.max(self(input_data),1)
                prediction = self(input_data)
                # print(prediction)

                loss = loss_fn(prediction, labels.long())

                val_loss += loss
                _, predicted = torch.max(prediction, 1)
                correct = labels == predicted
                total_correct += sum(correct.int())

        num_val_samples = len(eval_loader)
        avg_val_loss = val_loss / num_val_samples
        accuracy = total_correct / num_val_samples
        print(avg_val_loss)
        print(accuracy)
        return avg_val_loss.item(), accuracy.item()


def train(num_epochs, model, train_loader, optimizer, loss_fn, val_loader):
    # Initialize empty lists for tracking training and validation losses
    train_losses, val_losses = [], []
    train_losses_epoch, val_losses_epoch, val_accuracy_epoch = [], [], []
    val_loss, accuracy = model.evaluate(val_loader, loss_fn)

    # Print message to indicate beginning of training
    print("Begin training...")
    best_val_loss = np.inf
    patience = 5

    # Loop over number of epochs
    for epoch in range(1, num_epochs + 1):

        # Initialize variables for tracking loss
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_test_loss = 0.0
        train_loss = 0

        # Loop over training data
        for i, data in enumerate(train_loader):
            # Unpack input and output data
            inputs, outputs = data

            # Zero gradients
            optimizer.zero_grad()

            # Make prediction using model
            predicted_outputs = model(inputs)

            # Calculate loss
            train_loss = loss_fn(predicted_outputs, outputs.long())

            # Backpropagate loss to calculate gradients
            train_loss.backward()

            # Update model parameters based on gradients
            optimizer.step()

            # Update running total of training loss
            running_train_loss += train_loss.item()

            # Append current loss value to list of training losses
            train_losses.append(train_loss.item())

            # Append current loss value to list of validation losses
            val_losses.append(val_loss)

        # Calculate average training loss for current epoch
        train_loss_value = running_train_loss / len(train_loader)
        # val_loss = model.evaluate(val_loader, loss_fn)
        train_losses_epoch.append(train_loss_value)
        val_loss, val_acc = model.evaluate(val_loader, loss_fn)
        val_losses_epoch.append(val_loss)
        val_accuracy_epoch.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            # Reset the early stopping counter
            early_stopping_counter = 0
            print("New Best: ", val_loss)
        # Otherwise, increment the counter
        else:
            early_stopping_counter += 1
            # If the counter reaches the patience, stop the training
            if early_stopping_counter >= patience:
                print('Early stopping')
                break

        # Print statistics for current epoch
        print('Completed training batch', epoch, 'Training Loss is: %.4f' % train_loss_value, "/%.4f" % val_loss,
              'Training accuracy is: %.4f' % val_acc)
    return train_losses, val_losses, train_losses_epoch, val_losses_epoch, val_accuracy_epoch, epoch - 1

def mplot(data, label):

    # Set up the plot
    plt.figure()

    # Plot the running loss
    plt.plot(range(len(data)), data, label='Training loss')

    # Add labels to the plot
    plt.xlabel('Iteration')
    plt.ylabel(label)
    # Show the plot
    plt.show()

def plot_loss(losses,val_losses):
    # Set up the plot
    plt.figure()

    # Plot the running loss
    plt.plot(range(len(losses)), losses, label='Training loss')

    # Plot the validation loss
    plt.plot(range(len(val_losses)), val_losses, label='Validation loss')

    # Add labels to the plot
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


main()
