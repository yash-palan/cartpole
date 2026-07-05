import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import class_defns.classical_head_classes as chc
import class_defns.quantum_circuit_classes as qcc
import class_defns.RL_classes as rlc
import numpy as np
# import copy
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import pennylane as qp
import torch.nn.functional as F 
from train_file import initializing_qvc,creating_network
from class_defns.utility_functions import loading_dot_pt_files,plotting_testing_plot

def plot_rewards(rewards,mean_range=1):    
        # The parameter mean_range must allow equal separation of rewards
        if len(rewards) % mean_range != 0:
            raise ValueError
            
        # Calculate mean rewards
        mean_rewards = list()
        for i in range(round(len(rewards)/mean_range)):
            reward_on_range = rewards[i*mean_range:i*mean_range+mean_range]
            reward_on_range_mean = round(sum(reward_on_range)/len(reward_on_range))
            mean_rewards.append(reward_on_range_mean)
            
        plt.plot(range(len(mean_rewards)), mean_rewards)
        plt.show()



def testing_function(network,env,num_episodes,num_timesteps,buffer_size,batch,epsilon_val):
    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    one_iteration_size = 2*state_size+3

    # Defining the agent 
    # epsilon_decay = 1 makes it not decay over testing, since that is what we want
    dqn_agent = rlc.DQN_agent(state_size=state_size, 
                              action_size=action_size,
                              buffer_size=(buffer_size,one_iteration_size),
                              network= network,
                              batch_size = batch,
                              epsilon = epsilon_val,
                              epsilon_decay = 1)

    # Defining optimizer and loss function

    rewards, epsilon_values = list(), list() # Lists to keep logs of rewards and apsilon values, for plotting later
    time_step = 0 # Initalize timestep counter
    
    loop_start_time= time.time()
    random_seeds = torch.randint(low = 1, high = 10000  ,size =(num_episodes,))

    each_episode_each_iteration_reward = torch.zeros((num_episodes,num_timesteps))
    for episode in range(num_episodes):
        total_reward = 0
        # state, _ = env.reset()
        state, _ = env.reset(seed = random_seeds[episode].item()) 
        print(f'\nTesting on EPISODE {episode+1} with seed {random_seeds[episode].item()}')
        start = time.time()
        for time_val in range(num_timesteps):
            time_step += 1            
            state_tensor = torch.tensor(state,dtype=torch.float32,requires_grad=False)
            action = dqn_agent.epsilon_greedy_strategy(state_tensor)
            
            next_state, reward, terminal, truncated , _ = env.step(action)

            # next_state_tensor = torch.tensor(next_state,requires_grad=False)
            # complete_torch_tensor = dqn_agent.function_to_create_a_complete_torch_tensor(state_tensor, action, reward, next_state_tensor, terminal)
            # dqn_agent.replay_buffer.push_into_buffer(complete_torch_tensor)
            
            state = next_state
            total_reward += reward
            
            # Keeps track of the reward obtained in each iteration of each episode 
            each_episode_each_iteration_reward[episode,time_val] = total_reward

            if(terminal):
                print(f"Episode terminated at time:{time_val}/{num_timesteps}")
                break
        
        rewards.append(total_reward)
        epsilon_values.append(dqn_agent.epsilon)
                    
        # Print information about the Episode performed
        elapsed = time.time() - start
        print(f'Time elapsed during EPISODE {episode+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
    
    elapsed = time.time() - loop_start_time
    print(f'Total Time elapsed for training: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
    
    return([rewards, each_episode_each_iteration_reward])
    # plot_rewards(rewards)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # define some globals
    # num_episodes = 150
    num_episodes = 10
    # num_episodes = 10
    num_timesteps = 500
    batch_size = 64

    one_iteration_size = 2*state_size+3
    buffer_size = 5000

    # Creating the network
    # input_clip_learning_network = chc.neural_net(layer_geometry=torch.tensor([state_size,3],dtype=torch.int),
    #                                              activation_functions=nn.ModuleList([nn.Tanh()]))
    # # Creating the quantum variational circuit
    # qvc_network = initializing_qvc(number_of_layers=1,
    #                                 number_of_wires=3,
    #                                 quantum_function=qcc.complete_variational_quantum_circuit_function)
    
    # # Creating the tail of the circuit
    # understanding_qvc_network = chc.neural_net(layer_geometry=torch.tensor([3,64,action_size],dtype=torch.int),
    #                                              activation_functions=nn.ModuleList([nn.ReLU(),nn.Identity()]))

    # # Combining the three
    # network = creating_network(input_clip_learning_network,qvc_network,understanding_qvc_network)

    # Extracting parameters
    base_path = os.getcwd()
    # complete_path = base_path+'/results/ansatz 1 qubit 3/ansatz_1_200_run/'
    complete_path = base_path+'/results/ansatz 2 qubit 3/'
    # complete_path = base_path+'/results/'
    filename = "ansatz_1.pt"
    # network_state_dict = loading_dot_pt_files(complete_path+filename)
    # network.load_state_dict(network_state_dict)

    network = loading_dot_pt_files(complete_path+filename)
    network.eval()

    rewards, each_iteration_tensor = testing_function(network,env,num_episodes,num_timesteps,buffer_size,batch_size,epsilon_val=0.01)
    plotting_testing_plot(each_iteration_tensor)


