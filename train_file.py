import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
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
#################################
#################################

def save_network(filename,network):
    print(network.state_dict())
    torch.save(network.state_dict(),filename)
    return

#################################
#################################
def creating_network(input_clip_learning_network,qvc_network,understanding_qvc_network):
    network = rlc.agent_brain(input_clip_learning = input_clip_learning_network,
                              qvc = qvc_network,
                              understanding_qvc_nn= understanding_qvc_network)
    return network
#################################
#################################
def initializing_qvc(number_of_layers,number_of_wires,quantum_function):
    
    complete_weight_matrix = torch.rand(size=(number_of_layers,number_of_wires,3) ,dtype=torch.float32)
    # input_vector = torch.tensor([[0.1,0.2,0.3,0.4]],dtype=torch.float32)

    dev = qp.device('default.qubit',wires = number_of_wires)
    # qvc_func = complete_variational_quantum_circuit_function


    qvc_object = qcc.complete_quantum_variational_circuit(qvc = quantum_function,
                                                          complete_weight_matrix=complete_weight_matrix,
                                                          dev = dev)
    return(qvc_object)
#################################
#################################
if __name__=="__main__":
    env = gym.make('CartPole-v1')

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # define some globals
    # num_episodes = 150
    num_episodes = 150
    num_timesteps = 500


    one_iteration_size = 2*state_size+3
    buffer_size = 5000

    number_wires = 2
    number_layers = 2

    # Define the geometry of the system and saving it in a json
    config = {
        "input_clip_learning_network":
        {
            "layer_geometry":[int(state_size),int(number_wires)],
            "activation_functions":["Tanh"] 
        },
        
        "qvc_network":
        {
            "number_of_layers":number_layers,
            "number_of_wires":number_wires,
        },

        "understanding_qvc_network":
        {
            "layer_geometry":[int(number_wires),64,int(action_size)],
            "activation_functions":["ReLU","Identity"]
        }
    }

    activation_map = {
        "ReLU": nn.ReLU,
        "Identity": nn.Identity,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "LeakyReLU": nn.LeakyReLU,
    }

    base_path = os.getcwd()
    complete_path = base_path + '/results/'
    with open(complete_path+"config.json", "w") as f:
        json.dump(config, f, indent=4)



    
    # Creating the head
    # input_clip_learning_network = chc.neural_net(layer_geometry=torch.tensor([state_size,number_wires],dtype=torch.int),
    #                                              activation_functions=nn.ModuleList([nn.Tanh()]))
    input_clip_learning_network = chc.neural_net(layer_geometry=torch.tensor(config["input_clip_learning_network"]["layer_geometry"],dtype=torch.int),
                                                activation_functions=nn.ModuleList(
                                                activation_map[name]() for name in config["input_clip_learning_network"]["activation_functions"]
                                                    )
                                                )
    
    # Creating the quantum variational circuit

    # qvc_network = initializing_qvc(number_of_layers=number_layers,
    #                                 number_of_wires=number_wires,
    #                                 quantum_function=qcc.complete_variational_quantum_circuit_function)
    
    qvc_network = initializing_qvc(number_of_layers=config["qvc_network"]["number_of_layers"],
                                    number_of_wires=config["qvc_network"]["number_of_wires"],
                                    quantum_function=qcc.complete_variational_quantum_circuit_function)
    # Plotting the qvc
    random_input_vector = torch.rand(size=(1,number_wires))
    qvc_network.draw_quantum_circuit(input_vector=random_input_vector)
    # Creating the tail of the circuit
    # understanding_qvc_network = chc.neural_net(layer_geometry=torch.tensor([number_wires,64,action_size],dtype=torch.int),
    #                                              activation_functions=nn.ModuleList([nn.ReLU(),nn.Identity()]))
    understanding_qvc_network = chc.neural_net(layer_geometry=torch.tensor(config["understanding_qvc_network"]["layer_geometry"],dtype=torch.int),
                                                activation_functions=nn.ModuleList(
                                                activation_map[name]() for name in config["understanding_qvc_network"]["activation_functions"]
                                                    )
                                                )
    # input_clip_learning_network = chc.neural_net(layer_geometry=torch.tensor([4,64,64,2],dtype=torch.int),
    #                                              activation_functions=nn.ModuleList([nn.ReLU(),nn.ReLU(),nn.Identity()]))
    # input_clip_learning_network = chc.neural_net(layer_geometry=torch.tensor([4,64,4,4,2],dtype=torch.int),
    #                                              activation_functions=nn.ModuleList([nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.Identity()]))
    
    # input_clip_learning_network = chc.neural_net(layer_geometry=torch.tensor([4,64],dtype=torch.int),
    #                                                 activation_functions=nn.ModuleList([nn.ReLU()]))
    # # understanding_qvc_network = chc.neural_net(layer_geometry=torch.tensor([64,64,2],dtype=torch.int),
    # #                                              activation_functions=nn.ModuleList([nn.ReLU(),nn.ReLU()]))
    # qvc_network = None
    # Creating the tail of the circuit
    # understanding_qvc_network = None

    # Combining the three
    network = creating_network(input_clip_learning_network,qvc_network,understanding_qvc_network)

    # Defining the agent 
    # dqn_agent = rlc.DQN_agent(state_size, action_size,buffer_size=(buffer_size,one_iteration_size))
    dqn_agent = rlc.DQN_agent(state_size=state_size, 
                              action_size=action_size,
                              buffer_size=(buffer_size,one_iteration_size),
                              network= network)

    # Defining optimizer and loss function
    optimizer = torch.optim.Adam(params = dqn_agent.main_network.parameters(),lr=0.001, betas=(0.9, 0.999))
    loss_function = F.mse_loss


    rewards, epsilon_values = list(), list() # Lists to keep logs of rewards and apsilon values, for plotting later
    time_step = 0 # Initalize timestep counter
    
    loop_start_time= time.time()
    for episode in range(num_episodes):
        total_reward = 0
        state, _ = env.reset()
        print(f'\nTraining on EPISODE {episode+1} with epsilon {dqn_agent.epsilon}')
        start = time.time()
        for time_val in range(num_timesteps):
            time_step += 1
            
            # Update Target Network every {dqn_agent.update_rate} timesteps
            if time_step % dqn_agent.update_rate == 0:
                dqn_agent.update_target_network()
            
            state_tensor = torch.tensor(state,dtype=torch.float32,requires_grad=False)

            action = dqn_agent.epsilon_greedy_strategy(state_tensor)
            
            next_state, reward, terminal, truncated , _ = env.step(action)
            next_state_tensor = torch.tensor(next_state,requires_grad=False)
            # complete_torch_tensor = dqn_agent.function_to_create_a_complete_torch_tensor(state_tensor, action, reward, next_state, terminal)
            complete_torch_tensor = dqn_agent.function_to_create_a_complete_torch_tensor(state_tensor, action, reward, next_state_tensor, terminal)

            dqn_agent.replay_buffer.push_into_buffer(complete_torch_tensor)
            
            state = next_state
            total_reward += reward

            if(terminal):
                print(f"Episode terminated at time:{time_val}/{num_timesteps}")
                break

            # This is impotant so that we move on to the next step of the enviornment evolution

            # if(dqn_agent.replay_buffer.current_buffer_size > dqn_agent.start_training and time_step%dqn_agent.update_rate):
            #     dqn_agent.training_loop(optimizer,loss_function)
            if(dqn_agent.replay_buffer.current_buffer_size > dqn_agent.start_training):
                dqn_agent.training_loop(optimizer,loss_function)
        
        rewards.append(total_reward)
        epsilon_values.append(dqn_agent.epsilon)
        
        # Everytime an episode is finished, update Epsilon value to a lower value
        if dqn_agent.epsilon > dqn_agent.epsilon_min:
            dqn_agent.epsilon *= dqn_agent.epsilon_decay
            
        # Print information about the Episode performed
        elapsed = time.time() - start
        print(f'Time elapsed during EPISODE {episode+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')

        # If the agent got a reward >499 in each of the last 10 episodes, the training is terminated
        if sum(rewards[-10:]) > 4990:
            print('Training stopped because agent has performed a perfect episode in the last 10 episodes')
            break
    
    elapsed = time.time() - loop_start_time
    print(f'Total Time elapsed for training: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
    
    def plot_rewards(mean_range=1):    
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
    plot_rewards()

    # Saving weight parameters using torch.save

    filename = "ansatz_1.pt"
    # torch.save(filename,network)
    torch.save(network,complete_path+filename)
#################################
#################################