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
    """
    Just a function for saving a neural network.

    Parameters
    ------------
    filename:  Name of the file to be stored (along with the path)
                string ("*.pt" file) 
    network: The network that needs to be stored. Should be a nn.Module inhereted class.

    """
    print(network.state_dict())
    torch.save(network.state_dict(),filename)
    return

#################################
#################################
def creating_network(input_clip_learning_network,qvc_network,understanding_qvc_network):
    """
    Just creates the complete agent's brain

    Parameters
    -------------
    input_clip_learning: This is classical neural net used to clip inputs that have range from [-inf, inf].
    However, this can also be used for other purposes as well, like it can be used as an encoder for the quantum circuit.

    quantum_variational_circuit: The quantum layer of the agent's brain.
    
    understanding_qvc: This is a classical Neural net that will interpret the outputs of the
    quantum_variational_circuit

    Return
    ------------
    network: 

    """

    network = rlc.agent_brain(input_clip_learning = input_clip_learning_network,
                              qvc = qvc_network,
                              understanding_qvc_nn= understanding_qvc_network)
    return network
#################################
#################################
def initializing_qvc(number_of_layers,number_of_wires,quantum_function):
    """
    This function just initializes the quantum variational circuit 

    Paramters
    --------------
    number_of_layers: The number of repeating layers in the quantum circuit

    number_of_wires: the input wires of the quantum circuit

    quantum_function: The quantum function for the quantum circuit
    """
    
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
def training_loop(env,config:dict,activation_map:dict, betas=(0.9, 0.999)):
    """
    Train a DQN agent on a Gymnasium-style environment using a hybrid
    classical/quantum network architecture.

    Builds a network from up to three optional components defined in
    `config`: a classical "input" encoder (input_clip_learning_network),
    a quantum variational circuit (qvc_network), and a classical
    "understanding" decoder (understanding_qvc_network). These are
    combined into a single network and wrapped in a DQN agent with an
    experience replay buffer. The agent is then trained for a fixed
    number of episodes and timesteps, using an epsilon-greedy policy
    and periodic target-network updates.

    During training, per-episode total rewards and epsilon values are
    logged. After training completes, a summary plot of rewards is
    saved to disk, and both the trained network's weights and the
    reward history are saved to files under `config["complete_path"]`.

    Args:
        config (dict): Configuration dictionary of the form:
        {
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
            },
            "seed":int(seed),  
            "number_of_episodes":num_episodes,
            "number_of_timesteps":num_timesteps,
            "buffer_size":buffer_size,
            "complete_path":complete_path
            }
            where 
            - "seed" (int): Random seed for reproducibility (torch, numpy, env).
            - "qvc_network" (dict or None): Config for the quantum variational
              circuit, with "number_of_wires" and "number_of_layers".
            - "input_clip_learning_network" (dict or None): Config for the
              classical input network, with "layer_geometry" and
              "activation_functions".
            - "understanding_qvc_network" (dict or None): Config for the
              classical output/decoder network, with "layer_geometry" and
              "activation_functions".
            - "number_of_episodes" (int): Number of training episodes.
            - "number_of_timesteps" (int): Max timesteps per episode.
            - "buffer_size" (int): Size of the replay buffer.
            - "complete_path" (str): Directory path for saving outputs
              (plots, model weights, reward logs).

        env: A Gymnasium-style environment exposing `observation_space`,
            `action_space`, `reset(seed=...)`, and `step(action)`.

        activation_map: dictionary that maps the activation_functions in the 
        config file to the appropriate module. It is of the form
        
                activation_map = {
                    "ReLU": nn.ReLU,
                    "Identity": nn.Identity,
                    "Sigmoid": nn.Sigmoid,
                    "Tanh": nn.Tanh,
                    "LeakyReLU": nn.LeakyReLU,
                }
        
        betas: Just the betsa for the Adam optimiser used in this.
                Default value is (0.9, 0.999)

    Returns:
        None. Saves a rewards plot (.png), the
        trained network's weights (.pt), and per-episode rewards (.npy)
        to `config["complete_path"]`.
    
    """
    seed = config["seed"]
    if(config["qvc_network"] is not None):
        number_wires = config["qvc_network"]["number_of_wires"]
    # number_layers =
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    num_episodes = config["number_of_episodes"]
    num_timesteps = config["number_of_timesteps"]
    buffer_size = config["buffer_size"]
    one_iteration_size = 2*state_size+3

    # Setting up seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed) 
    # random.seed(seed)

    # Define the geometry of the system and saving it in a json
    complete_path = config["complete_path"]


    # Creating the head
    if(config["input_clip_learning_network"] is not None):
        input_clip_learning_network = chc.neural_net(layer_geometry=torch.tensor(config["input_clip_learning_network"]["layer_geometry"],dtype=torch.int),
                                                    activation_functions=nn.ModuleList(
                                                    activation_map[name]() for name in config["input_clip_learning_network"]["activation_functions"]
                                                        )
                                                    )
    else:
        input_clip_learning_network = None
    
    # Creating the quantum variational circuit
    if(config["qvc_network"] is not None):    
        qvc_network = initializing_qvc(number_of_layers=config["qvc_network"]["number_of_layers"],
                                        number_of_wires=config["qvc_network"]["number_of_wires"],
                                        quantum_function=qcc.complete_variational_quantum_circuit_function)
        # Plotting the qvc
        random_input_vector = torch.rand(size=(1,number_wires))
        qvc_network.draw_quantum_circuit(input_vector=random_input_vector)
    else:
        qvc_network = None
    # Creating the tail of the circuit

    if(config["understanding_qvc_network"] is not None): 
        understanding_qvc_network = chc.neural_net(layer_geometry=torch.tensor(config["understanding_qvc_network"]["layer_geometry"],dtype=torch.int),
                                                    activation_functions=nn.ModuleList(
                                                    activation_map[name]() for name in config["understanding_qvc_network"]["activation_functions"]
                                                        )
                                                    )
    else:
        understanding_qvc_network = None

    # Combining the three
    network = creating_network(input_clip_learning_network,qvc_network,understanding_qvc_network)

    # Defining the agent 
    dqn_agent = rlc.DQN_agent(state_size=state_size, 
                              action_size=action_size,
                              buffer_size=(buffer_size,one_iteration_size),
                              network= network)

    # Defining optimizer and loss function
    # The optimiser and the loss function can be made more general, however, for simplicity, this is ignored for the moment.
    optimizer = torch.optim.Adam(params = dqn_agent.main_network.parameters(),lr=dqn_agent.learning_rate, betas=betas)
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
        # if sum(rewards[-10:]) > 4990:
        #     print('Training stopped because agent has performed a perfect episode in the last 10 episodes')
        #     break
    
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
        plt.savefig(complete_path+f"train_rewards_summary_seed_{seed}.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    plot_rewards()

    # Saving weight parameters using torch.save
    # filename = "ansatz_1.pt"
    filename = f"ansatz_1_seed_{seed}.pt"
    torch.save(network,complete_path+filename)

    # save rewards in a .npy file
    filename = f"rewards_per_episode_seed_{seed}.npy"
    np.save(complete_path+filename, np.array(rewards))
#################################
#################################
if __name__=="__main__":
    # define some globals
    # num_episodes = 150
    env = gym.make('CartPole-v1')

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    num_episodes = 150
    num_timesteps = 500

    buffer_size = 5000

    number_wires = 2
    number_layers = 2

    start = 10
    end =  15
    step = 1

    # seed = 5142
    base_path = os.getcwd()
    complete_path = base_path + '/results/'

    for seed in range(start,end,step):
        print(f"\nStarted seed:{seed}")
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
            },
            "seed":int(seed),  
            "number_of_episodes":num_episodes,
            "number_of_timesteps":num_timesteps,
            "buffer_size":buffer_size,
            "complete_path":complete_path
        }

        # config = {
        #     "input_clip_learning_network":
        #     {
        #         # "layer_geometry":[int(state_size),int(number_wires)]
        #         "layer_geometry":[int(state_size),int(number_wires),int(number_wires),64,int(action_size)],
        #         # "activation_functions":["Tanh"] 
        #         "activation_functions":["ReLU","ReLU","ReLU","Identity"] 
        #     },
            
        #     "qvc_network":None,

        #     "understanding_qvc_network":None,
        #     "seed":int(seed),  
        #     "number_of_episodes":num_episodes,
        #     "number_of_timesteps":num_timesteps,
        #     "buffer_size":buffer_size,
        #     "complete_path":complete_path
        # }

        activation_map = {
            "ReLU": nn.ReLU,
            "Identity": nn.Identity,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU,
        }


        with open(complete_path+f"config_{seed}.json", "w") as f:
            json.dump(config, f, indent=4)

        training_loop(env,config,activation_map)

#################################
#################################