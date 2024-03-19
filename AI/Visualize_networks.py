import gym
import numpy as np


import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
import torch
import MModules.model as mmodel

from mpl_toolkits.mplot3d import Axes3D


env = gym.make('BipedalWalker-v3', render_mode="human")
# env = gym.make('Pendulum-v1',render_mode="human")

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)




network_folder = 'Models/'
actor_files = [f for f in os.listdir(network_folder) if f.endswith('_actor.pt')]
actor_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
weights, biases, epochs = [], [], []
for file in actor_files:
    epoch = int(file.split('_')[0])
    model = mmodel.Actor(S_DIM, A_DIM, A_MAX)
    model.load_state_dict(torch.load(os.path.join(network_folder, file)))
    weight, bias = [], []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight.append(param.detach().numpy().flatten())
        elif 'bias' in name:
            bias.append(param.detach().numpy().flatten())
    weights.append(np.concatenate(weight))
    biases.append(np.concatenate(bias))
    epochs.append([epoch] * len(weights[0]))

# Plot the weights and biases
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

print('epochs: ',np.array(epochs)[:,1],'\n index:', range(len(weights)),'\n weights: ', weights)
indicies =[*range(len(weights[0]))]
epoch_mesh, index_mesh = np.meshgrid(np.array(epochs)[:,1], indicies, indexing='xy')
flat_epoch=epoch_mesh.flatten()
flat_index= index_mesh.flatten()

ax1.scatter(flat_epoch, flat_index, weights)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Weight Index')
ax1.set_zlabel('Weight Value')
ax1.set_title('Weights over time')

#ax2.scatter(epochs, range(len(biases)), biases)
#ax2.set_xlabel('Epoch')
#ax2.set_ylabel('Bias Index')
#ax2.set_zlabel('Bias Value')
#ax2.set_title('Biases over time')

plt.show()
