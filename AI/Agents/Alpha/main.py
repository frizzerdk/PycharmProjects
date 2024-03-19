import gym
import numpy as np
import torch
import os
import psutil
import gc
import time
import imageio
import itertools
import matplotlib.pyplot as plt

from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')

import train
import buffer
import utils as utils
import wandb_helper as logger
from utils import tic,toc

import wandb
#### META #####

do_offline = False
do_plot=do_offline
if do_offline:
    os.environ['WANDB_MODE'] = 'dryrun'
if do_plot:
    figure, axis = plt.subplots(2,1)

# Constants
MAX_EPISODES = 3000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
EXPLOIT_FREQ = 5
SEED = 42
RENDER_GAP = 10


M_CONFIGS =  {
    "learning_rate": [0.001, 0.0001, 0.0005],#, 0.01, 0.002],
    "batch_size": [128, 256,36],
    "batch_multi": [16, 32,128],
    "gamma": [ 0.99,0.995],
    "tau": [0.001],
    "weight_decay": [0, 0.0001,0.00001],
    "notes": "Try stuff",
    "max_episodes": MAX_EPISODES,
    "max_steps": MAX_STEPS,
    "max_buffer": MAX_BUFFER,
    "max_total_reward": MAX_TOTAL_REWARD,
    "exploit_freq": EXPLOIT_FREQ,
    "seed": SEED,
    "render_gap": RENDER_GAP
}





configs = utils.generate_combinations(M_CONFIGS)
for M_CONFIG in configs:
    print(M_CONFIG)
    # Initialize the wandb project
    run = wandb.init(project="AC_RL_BipedalWalker-v3_sweep",config=M_CONFIG, reinit=True)

    # Initialize the wandb helper logger
    mlog = logger.MyLogger()

    # Create the gym environment
    env = gym.make('BipedalWalker-v3', render_mode="rgb_array")

    # Get the state and action dimensions
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_MAX = env.action_space.high[0]

    # Print the state and action dimensions
    print('State Dimensions: ', STATE_DIM)
    print('Action Dimensions: ', ACTION_DIM)
    print('Action Max: ', ACTION_MAX)

    # Create a memory buffer and a trainer
    memory = buffer.MemoryBuffer(M_CONFIG['max_buffer'])
    trainer = train.Trainer(
        STATE_DIM,
        ACTION_DIM,
        ACTION_MAX,
        memory,
        batch_size=M_CONFIG['batch_size'],
        batch_multi=M_CONFIG['batch_multi'],
        learning_rate=M_CONFIG['learning_rate'],
        gamma=M_CONFIG['gamma'],
        tau=M_CONFIG['tau'],
        weight_decay=M_CONFIG['weight_decay']
    )

    # Load models and create logger hooks
    start = 0
    #trainer.load_models(start)

    logger_critic = logger.LoggerHook(trainer.critic, trainer.critic_optimizer, mlog=mlog, model_name="critic")
    logger_actor = logger.LoggerHook(trainer.actor, trainer.actor_optimizer, mlog=mlog, model_name="actor")
    logger_critic_target = logger.LoggerHook(trainer.target_critic, trainer.critic_optimizer, mlog=mlog, model_name="critic_target")
    logger_actor_target = logger.LoggerHook(trainer.target_actor, trainer.actor_optimizer, mlog=mlog, model_name="actor_target")



    # Create arrays to store rewards, times, and frames
    total_reward_history = []
    total_exploit_reward_history = []
    total_time_history = []
    frames = []

    for episode in range(start, M_CONFIG['max_episodes']):
        observation = env.reset(seed=episode)[0]
        print('EPISODE:', episode)
        sum_reward = 0
        steps = 0
        tic()

        for step in range(M_CONFIG['max_steps']):
            steps=step
            state = np.asarray(observation)

            # Choose action based on exploration or exploitation
            if episode % M_CONFIG['exploit_freq'] == 0:
                action = trainer.get_exploitation_action(state)
            else:
                action = trainer.get_exploration_action(state)

            # Take action in the environment
            new_observation, reward, done, truncated, info = env.step(action)
            sum_reward += reward

            if done:
                new_state = None
            else:
                new_state = np.asarray(new_observation)
                memory.add(state, action, reward, new_state)

            observation = new_observation

            # Optimize the trainer
            trainer.optimize()

            # Render the environment
            if episode % M_CONFIG['exploit_freq'] == 0 and episode % M_CONFIG['render_gap'] == 0:
                frames.append(env.render())

            # Check if episode is done
            weak_perfomance=sum_reward<30 and step>200
            if done or weak_perfomance:
                break

        # Calculate the mean runtime and print the results
        mean_runtime = toc() / steps
        print('Mean Time:', "%.2f" % (mean_runtime * 1000), 'ms   Steps:', steps, '  toc:', "%.2f" % toc(), 's', 'Total Reward:', "%.2f" % sum_reward)

        # Store the results in the history arrays
        total_reward_history.append(sum_reward)
        total_time_history.append(mean_runtime)

        # Log data to wandb
        mlog.log({
            "reward": sum_reward,
            "mean_runtime": mean_runtime * 1000,
            "steps": steps,
            "runtime": toc(),
            "episode":episode
        })

        # Perform exploitation actions
        if episode % M_CONFIG['exploit_freq'] == 0:
            total_exploit_reward_history.append(sum_reward)

            # Get state, action, and reward
            s1 = trainer.s1
            a1 = trainer.a1
            r1 = trainer.r1
            s2 = trainer.s2

            # Log critic and actor information
            logger_critic.forward(s1, a1)
            logger_actor.forward(s1)
            logger_critic_target.forward(s1, a1)
            logger_actor_target.forward(s1)

            tic()

            if episode % M_CONFIG['render_gap'] == 0:
                # Create a video of the environment
                if len(frames) > 1:
                    directory = "./videos"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    anim_file = directory + "/animation_ep{}.gif".format(episode)
                    imageio.mimsave(anim_file, frames, fps=30)
                    mlog.log({"video": wandb.Video(anim_file, fps=5, format="gif")})
                    print('Render time:', "%.2f" % (toc() ), 's')
                    mlog.log({
                        "exploit_reward": sum_reward,
                        "render_time": toc() * 1000
                    })
                    frames = []

        # Send the logs to wandb
        mlog.send_log()

        # Perform garbage collection
        gc.collect()

        # Save models every 100 episodes
        if episode % 100 == 0:
            memory.save_to_file("mem_ep_{}".format(episode))
            trainer.save_models(episode)

        if do_plot and episode % 5 == 0:
            # Clear the subplots


            # Plot the reward history
            axis[0].cla()
            axis[0].plot(total_reward_history)
            axis[0].set_title("Total Reward")

            # Plot the exploit reward history
            axis[1].cla()
            axis[1].plot(total_exploit_reward_history)
            axis[1].set_title("Exploit Reward")

            # Update the plot
            plt.show(block=False)
            plt.pause(0.02)
            plt.show(block=False)
            plt.pause(0.02)
    print('Completed Episodes')
    run.finish()
