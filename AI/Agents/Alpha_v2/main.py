import gym
import numpy as np
import torch
import os
import gc
import time
import imageio
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')

import train
import buffer_fast as buffer
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

# ----------------------------------------
# Set up the environment
# ----------------------------------------
# Constants
MAX_EPISODES = 3000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
EXPLOIT_FREQ = 5
SEED = 42
RENDER_GAP = 10

# Configs
M_CONFIGS =  {
    "learning_rate": [0.001,0.005, 0.0001 ],#, 0.01, 0.002], # learning rate
    "batch_size": [128, 256,32], # batch_size taken from replay buffer
    "batch_multi": [32,8,128], # batch_size * batch_multi = total batch size
    "gamma": [ 0.99,0.995,0.90], # discount factor
    "tau": [0.001,0.005,0.0002], # update rate for target network
    "weight_decay": [0.00001,0, 0.0001], # weight decay for optimizer
    "theta": [0.15,0.05,0.3], # Ornstein-Uhlenbeck process mean reversion
    "sigma": [0.2,0.4,0.1], # Ornstein-Uhlenbeck process volatility
    "actor_reg_coeff": [10,0.001,100], # regularization coefficient for actor
    "critic_reg_coeff": [1,0.1,10], # regularization coefficient for critic
    "sampling_alpha": [0.2,0.4,0.01], # alpha for prioritized experience replay
    "sampling_beta": [0.2,0.4,0.01], # beta for prioritized experience replay
    "notes": "Try stuff",
    "max_episodes": MAX_EPISODES,
    "max_steps": MAX_STEPS,
    "max_buffer": MAX_BUFFER,
    "max_total_reward": MAX_TOTAL_REWARD,
    "exploit_freq": EXPLOIT_FREQ,
    "seed": SEED,
    "render_gap": RENDER_GAP
}

# Override configs
M_CONFIGS["learning_rate"] = [0.001]
M_CONFIGS["batch_size"] = [256]
M_CONFIGS["batch_multi"] = [32]
M_CONFIGS["gamma"] = [0.992]
M_CONFIGS["tau"] = [0.001]
M_CONFIGS["weight_decay"] = [0.00001]
M_CONFIGS["theta"] = [0.15]
M_CONFIGS["sigma"] = [0.2]
M_CONFIGS["actor_reg_coeff"] = [10]
M_CONFIGS["critic_reg_coeff"] = [1]
M_CONFIGS["sampling_alpha"] = [0.5]
M_CONFIGS["sampling_beta"] = [0.1]


# Make iterable cofigs for sweeps
configs = utils.generate_combinations(M_CONFIGS,do_all=False)
# Uncomment to run a single config
#configs = configs[0:1]

#----------------------------------------
# Main loop
#----------------------------------------
for M_CONFIG in configs:
    torch.manual_seed(M_CONFIG['seed'])
    print(M_CONFIG)
    # Initialize the wandb project
    run = wandb.init(project="AC_RL_BipedalWalker-v3_Alpha_prio_sweep",config=M_CONFIG, reinit=True)
    wandb.define_metric("exploit_reward", summary="max")
    wandb.define_metric("exploit_reward", summary="mean")
    wandb.define_metric("reward", summary="mean")
    wandb.define_metric("reward", summary="max")

    # Create the gym environment
    if do_offline:
        env = gym.make('BipedalWalker-v3', render_mode="human")
    else:
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
    memory = buffer.PrioritizedMemoryBuffer(M_CONFIG['max_buffer'],
                                            state_shape=STATE_DIM,
                                            action_shape=ACTION_DIM,
                                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                            alpha=M_CONFIG['sampling_alpha'],
                                            beta=M_CONFIG['sampling_beta'])

    agent = train.Agent(
        STATE_DIM,
        ACTION_DIM,
        ACTION_MAX,
        memory,
        batch_size=M_CONFIG['batch_size'],
        batch_multi=M_CONFIG['batch_multi'],
        learning_rate=M_CONFIG['learning_rate'],
        gamma=M_CONFIG['gamma'],
        tau=M_CONFIG['tau'],
        weight_decay=M_CONFIG['weight_decay'],
        theta=M_CONFIG['theta'],
        sigma=M_CONFIG['sigma'],
        actor_reg_coeff=M_CONFIG['actor_reg_coeff'],
        critic_reg_coeff=M_CONFIG['critic_reg_coeff'],

    )

    # Load models and create logger hooks
    pretrain = False
    start = 0
    if pretrain:
        agent.load_models(start)
        memory.load_from_file("mem_ep_{}".format(start))

    logger_critic = logger.LoggerHook(agent.critic, agent.critic_optimizer, model_name="critic")
    logger_actor = logger.LoggerHook(agent.actor, agent.actor_optimizer, model_name="actor")
    logger_critic_target = logger.LoggerHook(agent.target_critic, agent.critic_optimizer, model_name="critic_target")
    logger_actor_target = logger.LoggerHook(agent.target_actor, agent.actor_optimizer, model_name="actor_target")



    # Create arrays to store rewards, times, and frames
    total_reward_history ,total_exploit_reward_history, total_time_history, frames = [],[],[],[]
    for episode in range(start, M_CONFIG['max_episodes']):

        print('EPISODE:', episode)
        tic()
        observation = env.reset(seed=episode)[0]
        sum_reward = 0
        steps = 0

        for step in range(M_CONFIG['max_steps']):
            steps=step
            state = np.asarray(observation)

            # Choose action based on exploration or exploitation
            if episode % M_CONFIG['exploit_freq'] == 0:
                action = agent.get_exploitation_action(state)
            else:
                action = agent.get_exploration_action(state)

            # Take action in the environment
            observation, reward, done, truncated, info = env.step(action)
            sum_reward += reward

            new_state = np.asarray(observation)
            memory.add(state, action, reward, new_state,done,truncated)

            # Optimize the trainer
            agent.step_done()

            # Render the environment
            if episode % M_CONFIG['exploit_freq'] == 0 and episode % M_CONFIG['render_gap'] == 0:
                frames.append(env.render())

            # Check if episode is done
            weak_perfomance=sum_reward<20 and step>300
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
        indices = memory.indicies.data.cpu()
        probabilities = memory.probabilities.data.cpu()

        mlog.log({'Priorities': memory.priorities[1:memory._len].data.cpu(),
                  'Sampled_priorities': memory.priorities[indices].data.cpu(),
                  'probabilities': probabilities,
                  'td_errors': memory.td_errors,
                  'Indicies': indices})

        # Perform exploitation actions
        if episode % M_CONFIG['exploit_freq'] == 0:
            total_exploit_reward_history.append(sum_reward)

            # Get state, action, and reward
            s1 = agent.s1
            a1 = agent.a1
            r1 = agent.r1
            s2 = agent.s2

            # Log critic and actor information
            logger_critic.forward(s1, a1)
            logger_actor.forward(s1)
            logger_critic_target.forward(s1, a1)
            logger_actor_target.forward(s1)

            tic()

            if episode % M_CONFIG['render_gap'] == 0 and not do_offline:
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
        mlog.log({},commit=True)

        # Perform garbage collection
        gc.collect()

        # Save models every 100 episodes
        if episode % 100 == 0:
            memory.save_to_file("mem_ep_{}".format(episode))
            agent.save_models(episode)

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