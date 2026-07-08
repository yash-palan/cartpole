import torch
import torch.nn as nn
import class_defns.classical_head_classes as chc
import class_defns.quantum_circuit_classes as qcc
import numpy as np
import copy
# import gymnasium as gym
import matplotlib.pyplot as plt
#################################
#################################
class agent_brain(nn.Module):
    """
    This class defines the brain of the agent, i.e. the main nn architecture that will 
    make decision. 

    Parameters
    ---------------
    input_clip_learning: This is classical neural net used to clip inputs that have range from [-inf, inf].
    However, this can also be used for other purposes as well, like it can be used as an encoder for the quantum circuit.

    quantum_variational_circuit: The quantum layer of the agent's brain.
    
    understanding_qvc: This is a classical Neural net that will interpret the outputs of the
    quantum_variational_circuit
    """
    def __init__(self,
                 input_clip_learning:chc.neural_net,
                 qvc:qcc.complete_quantum_variational_circuit,
                 understanding_qvc_nn:chc.neural_net):
        super().__init__()
        self.input_clip_learning = input_clip_learning
        self.quantum_variational_circuit = qvc
        self.understanding_qvc = understanding_qvc_nn

    def checks(self):
        """
        Here I just check if the patching of the three is actually correct, as in the 
        output of input_clip learning is the input for quantum_variational_circuit and so 
        on.
        """
        return
    
    def forward(self,input_vector):
        x1 = self.input_clip_learning.forward(input_vector=input_vector)
        
        if(self.quantum_variational_circuit is not None):
            x2 = self.quantum_variational_circuit.forward(input_vector=x1)
        else:
            x2 = x1
        
        if(self.understanding_qvc is not None):
            x3 = self.understanding_qvc.forward(input_vector=x2)
        else:
            x3 = x2
        return(x3)
    
    def extraction_of_qvc_output_states(self,input_vector):
        """
        A function that extracts the intermediate output of the quantum variational circuit.
        Useful for understanding of the quantum variational circuit beyond just the reward structure.

        Parameters
        --------------
        input_vector: The input vector to the quantum circuit
        """
        if(self.quantum_variational_circuit is None):
            raise Exception("There is some mistake. There is no quantum circuit in this network.")
        
        x1 = self.input_clip_learning.forward(input_vector=input_vector)
        x2 = self.quantum_variational_circuit.forward(input_vector=x1)
        return([x1,x2])
#################################
#################################
class Buffer_class:
    """
    Creates a buffer object. This is used to represent the memory 
    of the RL-agent.

    Parameters
    ---------------------
    buffer_size = (max_number_of_elements, number_of_input_vectors) 

    note here that self.buffer_size is only the max_number_of_elements in the tensor
    """
    def __init__(self,buffer_size:int):

        self.buffer = torch.zeros(size = buffer_size,requires_grad= False)
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size[0]
        self.pointer = 0

    def push_into_buffer(self,data:torch.Tensor):
        """
        Function to push new state data into the buffer.

        Parameter
        --------------------
        data: torch 1D tensor with length number_of_input_vectors. 
        This represents what the agent remembers. 

        """
        ptr_to_push_into = self.pointer % self.max_buffer_size
        self.buffer[ptr_to_push_into,:] = data
        self.pointer += 1
        if(self.current_buffer_size >= self.max_buffer_size):
            self.current_buffer_size = self.max_buffer_size
        else:
            self.current_buffer_size += 1
        return

    def sample_from_buffer(self,number_of_elements:int):
        """  
        Code for sampling from the buffer. This represents the agent's ability to 
        randomly remember old information on which it is trained.

        Parameters
        -------------
        number_of_elements: The number of elements that the agents wishes to remember.
                            data type: int
        """        
        if(number_of_elements>self.current_buffer_size):
            raise Exception("number of elements larger than current size of the buffer. Check again.")
        
        locations = np.random.choice(a=self.current_buffer_size,size = number_of_elements,replace=False)
        
        return(self.buffer[locations])
#################################
#################################
class DQN_agent:
    """
    This is the true agent class, which includes the brain as well as other
    impotant information for learning. 

    Parameters
    ---------------
    action_size: Defines the size of the action space (discrete for our case)
                dtype: int

    state_size: Defines the size of teh state space 
                dtype: int

    buffer_size: Defines the size of the buffer (memory)
                (max_number_of_elements, number_of_input_vectors) 
    
    network: Defines the agent's brain.
    
    Inputs in **kwargs
    epsilon: Starting value of epsilon for the epsilon greedy strategy of selecting next state

    epsilon_min = minimum value of epsilon for the epsilon greedy strategy of selecting next state
    
    gamma = weight given to the future Q values in the Bellman equation
    
    learning_rate = learning rate of the optimiser
    
    update_rate = Number of iterations after which the target network gets updated
    
    start_training = The iteration number after which the main network starts training. Added
                    so that we have enough elements in the buffer to sample from before starting training.
    
    batch_size = Defines the batch size of the mini batch sampled for training the main network
    
    epsilon_decay: Defines the decay of the epsilon paramter as a function of episodes, for the epsilon greedy
                    strategy. We go from exploration to exploitation.
    """
    def __init__(self,
                 state_size:int,
                 action_size:int,
                 buffer_size,
                 network:agent_brain,
                 **kwargs):
        self.action_size = action_size
        self.state_size = state_size

        self.epsilon = 1. if kwargs.get('epsilon') is None else kwargs.get('epsilon') 
        self.epsilon_min = 0.01 if kwargs.get('epsilon_min') is None else kwargs.get('epsilon_min')
        self.gamma = 0.98 if kwargs.get('gamma') is None else kwargs.get('gamma')
        self.learning_rate = 1e-3 if kwargs.get('learning_rate') is None else kwargs.get('learning_rate')
        self.update_rate = 10 if kwargs.get('update_rate') is None else kwargs.get('update_rate')
        self.start_training = 64 if kwargs.get('start_training') is None else kwargs.get('start_training')
        self.batch_size = 64 if kwargs.get('batch_size') is None else kwargs.get('batch_size')
        self.epsilon_decay = 0.98 if kwargs.get('epsilon_decay') is None else kwargs.get('epsilon_decay')

        self.replay_buffer = Buffer_class(buffer_size = buffer_size)

        # self.main_network = neural_net_class(inputs = state_size,outputs = action_size)
        self.main_network = network


        # Creating a frozen target network
        self.target_network = copy.deepcopy(network)
        self.freezing_target_network()
        self.update_target_network()

    def freezing_target_network(self):
        """
        Function for freezing the parameters of the target_network
        """
        for params in self.target_network.parameters():
            params.requires_grad = False
    
    def update_target_network(self):
        """
        Function to update the weights of the target_network
        """
        # self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network = copy.deepcopy(self.main_network)

    def epsilon_greedy_strategy(self,state:torch.Tensor):
        """
        Returns the number for the correct action.
        
        Note- That this only works for the case when you input ONE state and DOES NOT TAKE BATCHING INTO ACCOUNT.
        Needs to be rewritten to take into account batching. 
        Does not affect the code since this is not called in the training loop (on batches) and we only 
        call it every time we have ONE state.

        Parameters
        -------------
        state: 1D torch tensor which holds the current state of the evironment 
        """
        p = np.random.random()
        action_number = 0
        if(p<self.epsilon):
            action_number = np.random.randint(low=0,high = self.action_size )
        else:
            with torch.no_grad():
                action_number = torch.argmax(
                                        self.main_network.forward( input_vector=state.reshape((1,-1)) ).flatten() 
                                        ).item()
        return(action_number)
    
    def function_to_create_a_complete_torch_tensor(self,state:torch.Tensor, action, reward, next_state:torch.Tensor, terminal):
        """
        Simple utility function to convert a set of various vectors into one tensor
        
        """
        
        c = torch.zeros(size = (len(state)*2+3,),requires_grad=False)
        c[0:len(state)] = state
        c[len(state)] = action
        c[len(state)+1] = reward
        c[len(state)+2: len(state)+2+len(state)] = next_state
        c[len(state)+2+len(state)] = terminal
        return(c)

    def function_to_extract_each_size(self,data_set:torch.Tensor):
        """
        Just a function to extract each mini batches of elements from the buffer/memory.
        This is specific to the buffer defined in the code. The structure is
        (state, action, reward, next_state,terminal)

        Parameters:
        ------------------
        dataset: A randomly sampled set of from the buffer data
                torch tensor
        """
        state_batch = data_set[:,0:self.state_size]
        action_batch = data_set[:,self.state_size].to(dtype=torch.int32)
        reward_batch = data_set[:,self.state_size+1]
        next_state_batch = data_set[:,self.state_size+2:self.state_size+2+self.state_size]
        terminal_batch = data_set[:,self.state_size+2+self.state_size].to(dtype=torch.int)

        return(state_batch,action_batch,reward_batch, next_state_batch,terminal_batch) 

    def training_loop(self,optimizer,loss_function):
        """
        A single loop of the training cycle of the main network

        Parameters
        --------------
        optimiser: The optimiser used for optimisation of the main network

        loss_function: Loss function used for optimisation of the main network
        """
        data_set = self.replay_buffer.sample_from_buffer(number_of_elements=self.batch_size)
        state_batch,action_batch,reward_batch, next_state_batch,terminal_batch = self.function_to_extract_each_size(data_set)
        
        with torch.no_grad():
            reward_2 = reward_batch + self.gamma*torch.max(self.target_network.forward(next_state_batch),dim = 1).values
            # y = torch.where(terminal_batch>0.5, reward_batch, reward_2  )
            y = torch.where(terminal_batch>0.5, reward_batch, reward_2  )

        q_values = self.main_network.forward(state_batch)                # (batch, action_size)
        q_selected = torch.gather(input = q_values,dim= 1, index = action_batch.unsqueeze(1)).squeeze(1) # (batch,) — Q(s,a) for the action actually taken
        loss = loss_function(q_selected, y)

        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()




