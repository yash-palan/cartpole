import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import class_defns.RL_classes as rlc
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from class_defns.utility_functions import loading_dot_pt_files,plotting_testing_plot


def testing_function(network,env,num_episodes,num_timesteps,buffer_size,batch,epsilon_val):
    """
    Tests a DQN agent on a Gymnasium-style environment using a hybrid
    classical/quantum network architecture.

    Args:
        network: A agent_brain class type of network which encodes the 
        brain of the agent

        env: A Gymnasium-style environment exposing `observation_space`,
            `action_space`, `reset(seed=...)`, and `step(action)`.

        num_episodes: The number of episodes over which the testing has to happen
        
        num_timesteps

        buffer_size: buffer size of the memory. Useless here but necessary for creating 
        the agent. Can be any value.

        batch: batch_size for training, but is useless here. Need it for creating the
        agent.

        epsilon_val: The epsilon value for the greedy epsilon policy.

    Returns:
        [rewards, each_episode_each_iteration_reward]
    
    """
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


