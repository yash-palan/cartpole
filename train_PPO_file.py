"""
The file which holds the class for the PPO agent 
Author: Yash Palan
Date: 14 August, 2026
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import torch.nn as nn
import class_defns.classical_head_classes as chc
import class_defns.quantum_circuit_classes as qcc
import class_defns.PPO_RL_class as ppo_rlc
import numpy as np
import gymnasium as gym
import time
# import torch.nn.functional as F 
from train_file import save_network,creating_network,initializing_qvc
#################################
#################################
def function_for_creating_the_network(config:dict,activation_map:dict):
        # Creating the head
    if(config["input_clip_learning_network"] is not None):
        input_clip_learning_network = chc.neural_net(layer_geometry=torch.tensor(config["input_clip_learning_network"]["layer_geometry"],dtype=torch.int),
                                                    activation_functions=nn.ModuleList(
                                                    (activation_map[name]() if name !="Softmax" else activation_map[name](dim=-1)) for name in config["input_clip_learning_network"]["activation_functions"]
                                                        )
                                                    )
    else:
        input_clip_learning_network = None
    
    # Creating the quantum variational circuit
    if(config["qvc_network"] is not None):    
        number_wires = config["qvc_network"]["number_of_wires"]
        qvc_network = initializing_qvc(number_of_layers=config["qvc_network"]["number_of_layers"],
                                        number_of_wires=config["qvc_network"]["number_of_wires"],
                                        quantum_function=qcc.complete_variational_quantum_circuit_function,
                                        entanglement_coupling=config["entanglement_coupling"])
        # Plotting the qvc
        random_input_vector = torch.rand(size=(1,number_wires))
        qvc_network.draw_quantum_circuit(input_vector=random_input_vector
                                         ,to_save=config["to_save"]
                                         ,complete_file_name=config["complete_file_name"])
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

    # Combining the three into actor and critic networks
    actor = creating_network(input_clip_learning_network,qvc_network,understanding_qvc_network)
    return actor
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
    seed = config["hyperparameters"]["seed"]

    complete_path = config["complete_path"]

    actor = function_for_creating_the_network(config = config["actor_network"],activation_map=activation_map)
    critic = function_for_creating_the_network(config = config["critic_network"],activation_map=activation_map)

    # Defining the agent 
    ppo_agent = ppo_rlc.PPO_agent(  
                                actor_network=actor,
                                critic_network=critic,
                                env= env,
                                hyperparameters=config["hyperparameters"]
                                )
    
    loop_start_time = time.time()

    ppo_agent.train_function(optimiser_actor=torch.optim.Adam,optimiser_critic=torch.optim.Adam,)    

    elapsed = time.time() - loop_start_time

    filename = complete_path+f"train_rewards_summary_seed_{seed}"
    ppo_agent.plot_training_results(filename)

    print(f'Total Time elapsed for training: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
    

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

    number_wires = 3
    number_layers = 2

    start = 0
    end =  10
    step = 1

    # seed = 5142
    base_path = os.getcwd()
    # complete_path = base_path + '/results/'
    # complete_path = base_path + '/results_circular/'
    complete_path = base_path + '/results_PPO/'

    for seed in range(start,end,step):
        print(f"\nStarted seed:{seed}")

        to_save = False
        complete_file_name = None
        if(seed-start ==0):
            to_save = True
            complete_file_name = complete_path + "circuit_diagram.png"

        # config = {
        #     "input_clip_learning_network":
        #     {
        #         "layer_geometry":[int(state_size),int(number_wires)],
        #         "activation_functions":["Tanh"] 
        #     },
            
        #     "qvc_network":
        #     {
        #         "number_of_layers":number_layers,
        #         "number_of_wires":number_wires,
        #     },

        #     "understanding_qvc_network":
        #     {
        #         "layer_geometry":[int(number_wires),64,int(action_size)],
        #         "activation_functions":["ReLU","Identity"]
        #     },
        #     "seed":int(seed),  
        #     "number_of_episodes":num_episodes,
        #     "number_of_timesteps":num_timesteps,
        #     "buffer_size":buffer_size,
        #     "complete_path":complete_path,
        #     "entanglement_coupling":'circular',
        #     "to_save":to_save,
        #     "complete_file_name":complete_file_name
        # }

        config = {
            "actor_network":{
                "input_clip_learning_network":
                    {
                        # "layer_geometry":[int(state_size),int(number_wires),int(number_wires),64,int(action_size)],
                        # "activation_functions":["ReLU","ReLU","ReLU","Identity"] 
                        # "layer_geometry":[int(state_size),64,64,int(action_size)],
                        # "activation_functions":["ReLU","ReLU","Softmax"] 
                        "layer_geometry":[int(state_size),2,2,64,int(action_size)],
                        "activation_functions":["ReLU","ReLU","ReLU","Softmax"] 
                    },
                "qvc_network":None,
                "understanding_qvc_network":None
            },
            "critic_network":{
                "input_clip_learning_network":
                    {
                        # "layer_geometry":[int(state_size),int(number_wires),int(number_wires),64,1],
                        # "activation_functions":["ReLU","ReLU","ReLU","Identity"] 
                        
                        # "layer_geometry":[int(state_size),64,64,1],
                        # "activation_functions":["ReLU","ReLU","Identity"]
                        "layer_geometry":[int(state_size),2,2,64,1],
                        "activation_functions":["ReLU","ReLU","ReLU","Identity"]
                    },
                "qvc_network":None,
                "understanding_qvc_network":None
                },
            "hyperparameters":
            {
            "gamma": 0.98 
            ,"learning_rate_actor": 1e-3 
            ,"learning_rate_critic": 1e-3 
            ,"batch_size": 32
            ,"number_of_episodes": 150 
            ,"seed" : int(seed)  
            ,"max_time_per_trajectory": 500 
            ,"total_evo_time_over_all_trajectories": 2000 
            ,"num_of_workers": 0 
            ,"clip": 0.2  
            ,"epochs" : 5 
            ,"entropy_coef":0.00
            },
            "complete_path":complete_file_name
        }

        activation_map = {
            "ReLU": nn.ReLU,
            "Identity": nn.Identity,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU,
            "Softmax":nn.Softmax
        }


        with open(complete_path+f"config_{seed}.json", "w") as f:
            json.dump(config, f, indent=4)

        training_loop(env,config,activation_map)

#################################
#################################