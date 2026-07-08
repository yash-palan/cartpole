import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import class_defns.quantum_circuit_classes as qcc
from class_defns import RL_classes as rlc
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import pennylane as qp
from class_defns.utility_functions import loading_dot_pt_files,plotting_testing_plot
from sklearn.decomposition import PCA

def pca_dimension_extraction(data):
    """
    Function to extract the PCA dimension for the data. 
    This extract the number of components that encode
    95% of the information

    Parameter
    -------------
    data: np.array that holds the data on which PCA is performed
        size = (samples, features)

    Returns
    ------------
    dimension of the vector space that encodes 95% of the information
    in the data 
    """
    # 1. Sample data (100 samples, 5 features)
    X = data

    # 2. Initialize PCA and specify number of components
    pca = PCA(n_components=0.95)

    # 3. Fit the model and transform the data
    # Scikit-Learn automatically centers the data internally
    X_pca = pca.fit_transform(X)

    # 4. Extract key attributes
    components = pca.components_                # Principal axes (V matrix)
    # exp_var = pca.explained_variance_           # Eigenvalues / Variance per component
    # exp_var_ratio = pca.explained_variance_ratio_ # Percentage of variance explained

    return(components.shape[0])

def extraction_of_intermediate_states(network,env,num_episodes,buffer_size=5000,batch=64,epsilon_val=0.1):
    """
    
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

    loop_start_time= time.time()
    random_seeds = torch.randint(low = 1, high = 10000  ,size =(num_episodes,))
    input_state_list = []
    output_state_list = []
    for episode in range(num_episodes):
        state, _ = env.reset(seed = random_seeds[episode].item()) 
        print(f'\nTesting on EPISODE {episode+1} with seed {random_seeds[episode].item()}')

        state_tensor = torch.tensor(state,dtype=torch.float32,requires_grad=False)
        input_state, output_state = dqn_agent.main_network.extraction_of_qvc_output_states(state_tensor.reshape((1,-1)))
        input_state_list.append(input_state.detach().flatten().numpy())
        output_state_list.append(output_state.detach().flatten().numpy())
    
    elapsed = time.time() - loop_start_time
    print(f'Total Time elapsed for extraction: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
    
    return([np.array(input_state_list), np.array(output_state_list)])

def compression_percentage(filename,env,num_episodes):
    network = loading_dot_pt_files(filename)
    network.eval()

    input_state_list, output_state_list = extraction_of_intermediate_states(network,env,num_episodes)
    input_state_dimension = pca_dimension_extraction(input_state_list)
    output_state_dimension = pca_dimension_extraction(output_state_list)
    compression_percentage_val = (1-(output_state_dimension/input_state_dimension))*100
    print(f"The amount of compression is {compression_percentage_val}.")
    return(compression_percentage_val)

def entanglement_entropy(input_state_list,complete_weight_matrix,device):
    n_wires = complete_weight_matrix.shape[1]
    entanglement_data = []

    for input_vector in input_state_list:
        input_vector_entanglement_data = []

        for wire in range(n_wires):
            @qp.qnode(device = device, interface="torch")
            def circuit(input_vector,wire):
                # --- state preparation ---
                qp.AngleEmbedding(input_vector*torch.pi, wires=range(n_wires))

                qp.Barrier(wires=range(n_wires))
                # --- variational circuit ---
                final_result = qcc.complete_variational_quantum_circuit_function_entanglement(input_vector = input_vector,
                                    complete_weight_matrix= complete_weight_matrix,wire = wire)
                return(final_result)   
               
            val = circuit(input_vector,wire)
            input_vector_entanglement_data.append(val.detach().numpy())
        entanglement_data.append(np.array(input_vector_entanglement_data))

    return(np.array(entanglement_data))


def extracting_and_infering_from_entanglement_entropy(filename,env,num_episodes):
    network = loading_dot_pt_files(filename)
    network.eval()

    input_state_list, output_state_list = extraction_of_intermediate_states(network,env,num_episodes)

    device = qp.device(name ="default.qubit")
    output_state_entanglement = entanglement_entropy(input_state_list,network.quantum_variational_circuit.weights,device)

    mean_output_state_entanglement = torch.mean(torch.tensor(output_state_entanglement),dim = 0)
    variance_of_output_state_entanglement = torch.var(torch.tensor(output_state_entanglement),dim = 0)
    std_dev = torch.sqrt(variance_of_output_state_entanglement)
    print(f"The average entanglement across output states is {mean_output_state_entanglement}.")
    print(f"The standard deviation of entanglement across output states is {std_dev}.")

    return([mean_output_state_entanglement,std_dev])

if __name__ =="__main__":
    base_path = os.getcwd()
    # complete_path = base_path+'/results/ansatz 1 qubit 3/ansatz_1_200_run/'
    # complete_path = base_path+'/results/ansatz 2 qubit 3/'
    # complete_path = base_path+'/results/ansatz 2 qubit 2/random seed result/'
    complete_path = base_path+'/results/ansatz 2 qubit 1/seed 1/'

    # complete_path = base_path+'/results/'
    # filename = "ansatz_1.pt"
    filename = "ansatz_1_seed_0.pt"
    env = gym.make('CartPole-v1')

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    num_episodes = 50

    mean,std_dev = extracting_and_infering_from_entanglement_entropy(complete_path+filename,env,num_episodes)
    # compression_percentage(complete_path+filename,env,num_episodes)

