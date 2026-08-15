"""
The file which holds the class for the PPO agent 
Author: Yash Palan
Date: 14 August, 2026
"""
import torch
import numpy as np
import torch.nn as nn
from class_defns.RL_classes import agent_brain 
import gymnasium as gym
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

class Dynamic_dataset(Dataset):
    def __init__(self,transform=None,target_transform = None):
        """
        Note - the transform and target_transform functions do nothing for the moment.
        """
        self.x_data = None
        self.y_data = None
        transform=transform 
        target_transform=target_transform

    def update_data_set(self,X_data,Y_data)->None:
        """
        Function for updating the dataset values
        
        Parameters
        ------------------
        x_data = [batch_size, ....]
        y_data = [batch_size]

        """
        self.x_data = X_data
        self.y_data = Y_data

    def __len__(self)->int:
        """
        Function for getting the batch size of the x_data

        """
        length = 0
        if(self.x_data is not None):
            length = len(self.x_data)

        return length

    def __getitem__(self, idx):
        """
        Function to get the shuffled data from the dataset

        Parameters
        -------------
        idx: np.array|list [int]. Indexes for the location to be extracted

        """
        return self.x_data[idx]

class PPO_agent:
    """
    class defintion for the PPO based agent. Note that action
    space is disceet.
    """
    def __init__(self
                ,env:gym.envs
                ,actor_network:agent_brain
                ,critic_network:agent_brain
                ,hyperparameters:dict
                ):

        self.env = env

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        # self.action_size = action_size
        # self.state_size = state_size

        # Setting the actor and critic networks
        # Actor takes in the state and outputs the probability of the action
        # Architecture : [state_size,hidden layers, action size]
        # Input: [batch size, state size]
        self.actor = actor_network
        # Critic takes in the state and outputs the value of the state 
        self.critic = critic_network

        self.total_reward_all_trajectories = []
        # Set the hyperparameters for the PPO evolution
        self._hyperparameters_initialize(hyperparameters=hyperparameters)

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}
        return

    def _hyperparameters_initialize(self
                                    ,hyperparameters:dict
                                    )->None:
        self.gamma = 0.98 if hyperparameters.get('gamma') is None else hyperparameters.get('gamma')
        self.learning_rate_actor = 1e-3 if hyperparameters.get('learning_rate_actor') is None else hyperparameters.get('learning_rate_actor')
        self.learning_rate_critic = 1e-3 if hyperparameters.get('learning_rate_critic') is None else hyperparameters.get('learning_rate_critic')
        self.batch_size = 10 if hyperparameters.get('batch_size') is None else hyperparameters.get('batch_size')
        # self.epsilon_decay = 0.98 if hyperparameters.get('epsilon_decay') is None else hyperparameters.get('epsilon_decay')
        self.num_episodes = 150 if hyperparameters.get('number_of_episodes') is None else hyperparameters.get('number_of_episodes')
        self.seed = None  if hyperparameters.get('seed') is None else hyperparameters.get('seed')
        self.max_time_per_trajectory = 500 if hyperparameters.get('max_time_per_trajectory') is None else hyperparameters.get('max_time_per_trajectory')
        self.total_evo_time_over_all_trajectories = 5000 if hyperparameters.get('total_evo_time_over_all_trajectories') is None else hyperparameters.get('total_evo_time_over_all_trajectories')
        self.num_of_workers = 1 if hyperparameters.get('num_of_workers') is None else hyperparameters.get('num_of_workers')
        self.clip = 0.2  if hyperparameters.get('clip') is None else hyperparameters.get('clip')
        # self.betas = (0.9,0.990) if hyperparameters.get('betas') is None else hyperparameters.get('betas')
        self.epochs = 5 if hyperparameters.get('epochs') is None else hyperparameters.get('epochs')
        self.entropy_coef = 0.05 if hyperparameters.get('entropy_coef') is None else hyperparameters.get('entropy_coef')

        
    def train_function(self
                       ,optimiser_actor=torch.optim.Adam
                       ,optimiser_critic= torch.optim.Adam
                        )->None:
        """
        Parameters
        ----------------
        optimiser_actor
        optimiser_critic
        """
        if(self.seed is not None):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.env.reset(seed=self.seed)
        else:
            self.env.reset()

        # Set up the optimiser for the actor and the critic
        actor_optim = optimiser_actor(params= self.actor.parameters()
                                      ,lr=self.learning_rate_actor
                                    #   ,betas=self.betas
                                      )
        critic_optim = optimiser_critic(params= self.critic.parameters()
                                        ,lr=self.learning_rate_critic
                                        # ,betas=self.betas
                                        )

        # Creating a torch data set which can be sent to
        # a data loader. This dataset also allows for
        # the data inside to be changed, thus allowed better 
        # performance. We create a dataset for both the actor
        # and the critic
        dataset_complete = Dynamic_dataset()


        # loader = DataLoader(dataset_complete
        #                     ,batch_size=self.batch_size
        #                     ,num_workers=self.num_of_workers
        #                     ,shuffle=True)
        
        for k in range(self.num_episodes):
            print(f'\nStarted EPISODE {k}/{self.num_episodes}')
            # Step 3-5 of the algorithm
            start_time_episode = time.time()
            with torch.no_grad():
                # Just to avoid any calculations that may by
                # fault have gradient on inside it
                trajectories = self.collect_trajectories_based_on_current_policy()

            elapsed=time.time()-start_time_episode
            print(f"-----Time to collect trajectories:{elapsed:.4f} secs= {round(elapsed/60, 3)} minutes")
            # data = (s_t​,a_t​,log(π_{θ_k}​​(a_t​∣s_t​),A_t​,R_t)
            torch_tensor_trjectories = torch.tensor(
                                                    np.array(trajectories)
                                                    ,dtype= torch.float32
                                                    ,requires_grad=False
                                                    )
            # print(f"-----Ran trajectories on current policy and extracted data")
            print(f"-----number of trajectories:{torch_tensor_trjectories.shape[0]}")

            print(f"-----reward: min {np.min(self.total_reward_all_trajectories[-1])}, max {np.max(self.total_reward_all_trajectories[-1])}")

            # Just for in place advantage normalisation
            # We have requires grad false but just for safety
            with torch.no_grad():
                # Note, this needs change when 
                advantage_col =  torch_tensor_trjectories[:,self.state_size+2]
                advantage_col_mean = advantage_col.mean()
                advantage_col_std = advantage_col.std()
                advantage_col.sub_(advantage_col_mean).div_(advantage_col_std+1e-10)

            # print(f"-----Normalized Advantage")

            # Create the dataset for the
            dataset_complete.update_data_set(X_data=torch_tensor_trjectories,
                                          Y_data=None)

            # Loader is created here since for shuffle I need some data inside
            # create the combined data loaders for both the actor and 
            # the critic, since we will need to update both the critic and 
            # actor simultaneously
            loader = DataLoader(dataset_complete
                        ,batch_size=self.batch_size
                        ,num_workers=self.num_of_workers
                        ,shuffle=True)
            
            for epoch in range(self.epochs):
                # This is step 6-7 in the algorithm
                single_epoch_start_time = time.time()
                self.train_single_step(
                                    actor_optim
                                    ,critic_optim
                                    ,data_loader_combined=loader
                                    )
                elapsed =time.time()-single_epoch_start_time
                print(f"-----Finished epoch {epoch}/{self.epochs} of optimisation: {elapsed:.2f} secs= {round(elapsed/60, 3)} minutes")

            elapsed = time.time()-start_time_episode

            print(f'Time elapsed during EPISODE {k}/{self.num_episodes}: {elapsed:.2f} seconds = {round(elapsed/60, 3)} minutes')

            # print(f"\nFinished episode {k}/{self.num_episodes} in time:{()} sec = {(time.time()-start_time_episode)/60} min")
        return 

    def train_single_step(self
                        ,optimiser_actor
                        ,optimiser_critic
                        ,data_loader_combined:DataLoader):
        
        for batch_x in data_loader_combined:
            optimiser_actor.zero_grad()
            optimiser_critic.zero_grad()

            # batch_x data = (s_t​,a_t​,log(π_{θ_k}​​(a_t​∣s_t​),A_t​,R_t)
            # This is a problem. CHeck this again

            state_data,action_data,batch_log_probs,A_k,batch_rtgs = self.extract_data_from_dataset(batch_x)
            

            # Computing the value function V and log probs based on current
            # policy \pi_{\theta} and V_{\phi}
            V,curr_log_probs,entropy =self.evaluate_data_on_current_policy(
                                                        state_data=state_data
                                                        ,action_data=action_data)

            ratios = torch.exp(curr_log_probs - batch_log_probs)
            surr1 = ratios * A_k          # <-- fixed A_k reused every epoch
            # torch.clamp is for enforcing the constraints
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k


            # actor_loss = (-torch.min(surr1, surr2)).mean()
            actor_loss = (-torch.min(surr1, surr2)).mean() - self.entropy_coef * entropy
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            actor_loss.backward()
            critic_loss.backward()

            optimiser_actor.step()
            optimiser_critic.step()

            self.logger['actor_losses'].append(actor_loss.detach())


        # return
    def extract_data_from_dataset(self,data):
        """
        Function to extract data from dataset
        """
        # Now allows for differnt state sizes
        state_data = data[:,0:self.state_size]
        # Note that since we have discreet actions, we only choose
        # the action number which is why the action is an int
        action_data = data[:,self.state_size].to(dtype = torch.int)

        batch_log_probs = data[:,self.state_size+1]
        A_k = data[:,self.state_size+2]
        batch_rtgs = data[:,self.state_size+3]

        return(state_data,action_data,batch_log_probs,A_k,batch_rtgs)

    def evaluate_data_on_current_policy(self
                                        ,state_data:torch.Tensor
                                        ,action_data:torch.Tensor):
        """
        
        """
        # action_probs dim: [batch_size, actions]
        action_probs = self.actor.forward(state_data)

        # Since, action_probs dim: [batch_size, actions]
        # log_prob_action = torch.log(action_probs[:,action_data])


        dist = Categorical(action_probs)
        # This directly gives the log probability for the actions_data values
        # by creating the distribution and then taking the correct actions from the
        # action_data
        log_prob_action = dist.log_prob(action_data)
        entropy = Categorical(action_probs).entropy().mean()

        # The below one is an alternative of the above two lines
        # log_prob_action = torch.log(action_probs[np.arange(action_probs.shape[0]),action_data])

        # The squeeze if to make the above from (batch_size,1)->(batch_size,)
        # The -1 is to fix for the case when batch_size = 1. Then it will squeeze 
        # comlpetely breaking the loop. But this way,
        # value_functions = self.critic.forward(state_data).squeeze()
        value_functions = self.critic.forward(state_data).squeeze(-1)

        return value_functions, log_prob_action, entropy

    def collect_trajectories_based_on_current_policy(self)->list[np.ndarray]:
        """
        Function that runs trajectories based on random intial states 
        based on the current policy \\pi_{\\theta} and value function
        \\V_{\\phi}. 

        Returns
        ------------
        list of type np.ndarray: This stores the outputs of the results 
        of the trajectory evolution in the list. Every element of the list
        is of the form 
        (state,action,log_prob_of_action,advatage,rewards_to_go)

        Note that for the code we have only discreet action. 
        """
        # Stores the data each episode in the each trajectory
        # data = (s_t​,a_t​,log(π_{θ_k}​​(a_t​∣s_t​),A_t​,R^t​)
        complete_data_list = []

        time_val = 0
        trajectory_number = 0
        total_reward_per_trajectory = []
        while(time_val <= self.total_evo_time_over_all_trajectories):
            # To start with a random state we reset the env
            state, _ = self.env.reset()
            total_reward = 0
            # trajectory_number = 0

            # What all we need for each episode in the trajectory
            # (s_t​,a_t​,log(π_{θ_k}​​(a_t​∣s_t​),A_t​,R^t​)
            # However, note that computing A_t, R^t needs the whole trajectory
            # So, one way to do this is rather to do this for each trajectory in the rollout
            # Thus, we compute A_t and R_t for each trajectory time t after the run of the trajectory and then 
            # add that into the data as needed

            state_data_per_trajectory = []
            action_data_per_trajectory = []
            log_prob_data_per_trajectory = []
            rewards_data_per_trajectory = []
            Value_function_data_per_trajectory = []
            for episode_time in range(self.max_time_per_trajectory) :

                state_data_per_trajectory.append(state)
                state_tensor = torch.tensor(
                                            state,
                                            dtype=torch.float32,
                                            requires_grad=False
                                            )
                # We have the torch.tensor since teh state input must be a tensor
                action, log_prob = self.get_action(state_tensor)   
                action_data_per_trajectory.append(action) 
                log_prob_data_per_trajectory.append(log_prob)

                value = self.get_critic(state_tensor)
                Value_function_data_per_trajectory.append(value)

                next_state, reward, terminal, truncated , _ = self.env.step(action)
                rewards_data_per_trajectory.append(reward)

                state = next_state
                total_reward += reward
                time_val += 1
                # trajectory_number += 1

                if(terminal or truncated):
                    break
            trajectory_number += 1
            # For CartPole, we are interested in this
            total_reward_per_trajectory.append(total_reward)

            # Now we run a loop over the trajectory to extract A_t and R_t
            reward_to_go = np.zeros(len(state_data_per_trajectory))

            if terminal:
                bootstrap_value = 0.0
            else:
                # loop ended via truncation, or hit max_time_per_trajectory without either flag set
                next_state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False)
                bootstrap_value = self.get_critic(next_state_tensor)

            for i in range(len(state_data_per_trajectory)-1,-1,-1,):
                current_reward = rewards_data_per_trajectory[i]

                if(i== len(state_data_per_trajectory)-1):
                    reward_to_go_at_t_plus_1 = bootstrap_value
                else:
                    reward_to_go_at_t_plus_1 = reward_to_go[i+1]

                current_value_function = Value_function_data_per_trajectory[i]

                current_reward_to_go = self.compute_rewards_to_go(
                                                            current_reward
                                                           ,reward_to_go_at_t_plus_1)
                # Need to store the rewards to go in the array so that the future time 
                # ones are accessible
                reward_to_go[i] = current_reward_to_go
                advantage_val = self.compute_advantage(reward_to_go=current_reward_to_go
                                                       ,value_function=current_value_function)

                # Check if complete_data_list has the correct shape
                complete_data_list.append(
                    np.concatenate(
                        (
                        state_data_per_trajectory[i]
                        ,np.array([action_data_per_trajectory[i]])
                        ,np.array([log_prob_data_per_trajectory[i]])
                        ,np.array([advantage_val])
                        ,np.array([current_reward_to_go])
                        ), axis = 0
                        )
                    )
        self.total_reward_all_trajectories.append(total_reward_per_trajectory)
        return complete_data_list

    def get_action(self,state:torch.Tensor)->list[int,float]:
        """
        This function takes in the state and outputs the action from the probabilities
        Note that this is only for one single state and not more states 
        Parameters
        -------------
        state: the state in which the env is currently

        Returns
        -------------
        [action,log_prob]
        """
        if len(state.shape)==1:
            new_state = torch.unsqueeze(state,dim=0)
        elif(state.shape[0]>1):
            raise Exception("Just need ONE state input.")
        else:
            new_state = state
        
        with torch.no_grad():
            # action_probs size = [batch_size, actions]
            action_probs = self.actor.forward(new_state)

            # Since we have just one state, so batch_size = 1
            cat_dist = Categorical(action_probs[0,:])

            # Since we need to pass the action as an int and NOT
            # a tensor to the action_probs
            action = cat_dist.sample().item()

            # Just check if this is truly the log
            log_prob_action = torch.log(action_probs[0,action]).item()

        # We have .item() since we wish to return just the value and not the tensor
        return([action,log_prob_action])

    def get_critic(self,state):
        """
        Returns the value of the critic (Value function) for the state give by
        "state" using the critic network.

        Parameters
        ---------------
        critic: 

        """
        with torch.no_grad():
            value = self.critic.forward(state)

        # We wish to get back the value and NOT the tensor
        return value.item()

    def compute_rewards_to_go(self
                              ,current_reward:float
                              ,reward_to_go_at_t_plus_1:float)->float:
        """
        Function to compute rewards to go. 
        Made this way so it can be changed later on
        without affecting the code.
        """
        return(current_reward+self.gamma*reward_to_go_at_t_plus_1)

    def compute_advantage(self
                          ,reward_to_go
                          ,value_function):
        """
        Function to compute the advantage. 
        Made this way so it can be changed later on
        without affecting the code.
        """
        return(reward_to_go-value_function)

    def plot_training_results(self,filename,show=True):
        """
        filename without the .npy
        """
        # Note the following should mathch self.num_episodes
        total_number_of_episodes = len(self.total_reward_all_trajectories)

        mean_data_all_updates = np.zeros((total_number_of_episodes,))
        std_data_all_updates = np.zeros((total_number_of_episodes,))
        for episode in range(total_number_of_episodes):
            trajectory = self.total_reward_all_trajectories[episode]
            tensor_trajectory = torch.tensor(trajectory,requires_grad=False)
            mean_data_all_updates[episode] = tensor_trajectory.mean().item()
            std_data_all_updates[episode] = tensor_trajectory.std().item()

        plt.figure()
        plt.plot(np.arange(total_number_of_episodes),mean_data_all_updates)
        plt.fill_between(np.arange(total_number_of_episodes), mean_data_all_updates-std_data_all_updates, mean_data_all_updates+std_data_all_updates
                         , facecolor='magenta', interpolate=True,alpha=0.2)

        # plt.savefig(complete_path+f"train_rewards_summary_seed_{seed}.png")
        plt.savefig(filename+f".png")
        if show:
            plt.show(block=False)
            plt.pause(3)
        plt.close()

        # save rewards in a .npy file
        # filename = f"rewards_per_episode_seed_{seed}.npy"
        np.save(filename+"_mean_data.npy", mean_data_all_updates)
        np.save(filename+"_std_data.npy", std_data_all_updates)
        
        return

    def save_weights(self,filename):
        """
        filename: Complete filename without the .pt label
        """
        # Saving weight parameters using torch.save
        # filename = "ansatz_1.pt"
        # filename = f"ansatz_1_seed_{seed}"
        torch.save(self.actor,filename+"_actor.pt")
        torch.save(self.critic,filename+"_critic.pt")

