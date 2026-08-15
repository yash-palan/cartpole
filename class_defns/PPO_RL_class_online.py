import torch
import torch.nn as nn
import numpy as np
from class_defns.RL_classes import agent_brain
from class_defns.classical_head_classes import neural_net
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import time
# class PPO_agent():
#     def __init__(self,
#                 state_size:int,
#                 action_size:int,
#                 buffer_size,
#                 actor:agent_brain,
#                 critic: agent_brain,
#                 **kwargs):

#         self.action_size = action_size
#         self.state_size = state_size

#         # Hyperparameters to be tuned
#         self.epsilon = 1. if kwargs.get('epsilon') is None else kwargs.get('epsilon') 
#         self.epsilon_min = 0.01 if kwargs.get('epsilon_min') is None else kwargs.get('epsilon_min')
#         self.gamma = 0.98 if kwargs.get('gamma') is None else kwargs.get('gamma')
#         self.learning_rate = 1e-3 if kwargs.get('learning_rate') is None else kwargs.get('learning_rate')
#         self.update_rate = 10 if kwargs.get('update_rate') is None else kwargs.get('update_rate')
#         self.start_training = 64 if kwargs.get('start_training') is None else kwargs.get('start_training')
#         self.batch_size = 64 if kwargs.get('batch_size') is None else kwargs.get('batch_size')
#         self.epsilon_decay = 0.98 if kwargs.get('epsilon_decay') is None else kwargs.get('epsilon_decay')

#         # self.replay_buffer = Buffer_class(buffer_size = buffer_size)

#         # Output is based on the number of actions (here 2)
#         self.actor = actor
#         # Critic has an architecture that gives the Value function as an output, so just 1 output
#         self.critic = critic

#         # Creating a frozen target network
#         self.target_network = copy.deepcopy(network)
#         self.freezing_target_network()
#         self.update_target_network()



class PPO_agent():
    def __init__(self,
                state_size:int,
                action_size:int,
                env,
                **kwargs):
        self.env = env
        self.action_size = action_size
        self.state_size = state_size

        # Hyperparameters to be tuned
        self._init_hyperparameters(self)


        # Step 1 of PPO: Creation of actor and critic networks
        # Output is based on the number of actions (here 2)
        layer_geometry_actor = [state_size,64,64,action_size]
        actor_activation_functions = nn.ModuleList([nn.ReLU(),nn.ReLU(),nn.Identity()])
        layer_geometry_critic = [state_size,64,64, 1]
        critic_activation_functions = nn.ModuleList([nn.ReLU(),nn.ReLU(),nn.Identity()])

        self.actor = neural_net(layer_geometry=layer_geometry_actor,activation_functions=actor_activation_functions)

        # Critic has an architecture that gives the Value function as an output, so just 1 output
        self.critic = neural_net(layer_geometry=layer_geometry_critic,activation_functions=critic_activation_functions)


        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# This logger will help us with printing out summaries of each iteration
        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode
        self.gamma =0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005

    def learn(self, total_timesteps):

        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far

        while t_so_far < total_timesteps:
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate V_{phi, k}
            # V = self.evaluate(batch_obs)
            for _ in range(self.n_updates_per_iteration):

                V, _ = self.evaluate(batch_obs, batch_acts)

                # ALG STEP 5
                # Calculate advantage
                A_k = batch_rtgs - V.detach()
                # Normalize advantages
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                # Calculate pi_theta(a_t | s_t)
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k


                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

				# Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
            self._log_summary()

			# Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self):
        # This is a function that computes the run for a given policy
        # Since PPO is an on policy RL method, one needs to compute the 
        # Rewards to go for this policy.


        t_so_far = 0 # Timesteps simulated so far

        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        t=0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)
            
                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews) 
        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def get_action(self,obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)

        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs,batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs