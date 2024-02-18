# Source: https://www.youtube.com/watch?v=6Yd5WnYls_Y

import os
from typing import Any
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random
from collections import namedtuple, deque
from DDPG_parameters import *

class OUActionNoise(object): 
    # to produce a temporarily correlated noise centered around a mean
    # of zero
    
    def __init__ (self, mu, sigma = 0.15, theta = 0.2, dt = 1e-2, x0 = None):
        # the values in the initialization are from the DDPG paper
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset() # to reset the temporal correlation

    def __call__(self):
        # to override the call function
        # Example on how the override works: 
        # noise = OUActionNoise()
        # ourNoise = noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
        self.x_prev = x # that is how to create the temporal correlations
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer(object):
    # For the replay buffer, you can use a data structure called DQ
    # In this implementation, np arrays are used because they allow clean manipulation of data types
    def __init__(self, max_size, input_shape, n_actions):
        # def __init__(self, action_size, buffer_size, batch_size, seed):
        self.mem_size = max_size
        self.mem_cntr = 0 # memory counter
        # self.state_memory = np.zeros((self.mem_size, *input_shape)) # * means unpack a tuple
        # self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.state_memory = np.zeros((self.mem_size, input_shape)) 
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32) # to save the done flags from openAI gym env.

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size # first available position
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_ # new state
        self.terminal_memory[index] = 1 - done # to be suitable for the update (Balman) equation
        self.mem_cntr += 1 # number of experiences added to the buffer
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice (max_mem, batch_size) # selects batch_size number of indices from the range [0, max_mem)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, reward, new_states, terminal
    
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = "tmp/ddpg"):
        #def __init__(self, state_size, n_actions, seed, fc1_units=1000, fc2_units=1000):
        # input_dims: state size
        # fc1_dims: fc1_units = 1000 (from RL paper)
        # fc2_dims: fc2_units = 1000 (from RL paper)
        # beta: learning rate = 10^-5 (from RL paper)
        # n_actions = 1
        # name: this is the name of the network, it is important for saving the network later

        # self.agent = DDPGAgent(alpha=LR_ACTOR, beta=LR_CRITIC, 
        #                   input_dims=INPUT_DIMENSION, 
        #                   tau=TAU, env=None, gamma=GAMMA,
        #                   n_actions=1, max_size=BUFFER_SIZE, 
        #                   layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE,
        #                   batch_size=BATCH_SIZE)

        # self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
        #                             n_actions=n_actions, name='Critic')

        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        # The neural network
        #self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0]) # from the DDPG paper
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1) # initialize the weights
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1) # initialize the biases
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0]) # from the DDPG paper
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2) # initialize the weights
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2) # initialize the biases
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Note: because we have a normalized layer, we have to use train and eval functions later

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003 # from the DDPG paper
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3) # initialize the weights
        T.nn.init.uniform_(self.q.bias.data, -f3, f3) # initialize the biases

        self.optimizer = optim.Adam(self.parameters(), lr = beta) # Note: self.parameters is inherited from nn.Module
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state, action):
        # First layer
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        # Second layer
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        # what about the regularizer? 
        # the DDPG paper uses L2 regularizer with a value of 0.01

        return state_action_value
    
    def save_checkpoint(self):
        print(' ... Saving Checkpoint ... ')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(' ... Loading Checkpoint ... ')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
    #def __init__(self, state_size, n_actions, seed, fc1_units=1000, fc2_units=1000):
        # alpha: learning rate = 10^-5 (from RL paper)
        # input_dims: state size
        # fc1_dims: fc1_units = 1000 (from RL paper)
        # fc2_dims: fc2_units = 1000 (from RL paper)
        # Note: you can add action_bound variable that is used in the output layer to bound the output
        # Note: action_bound is positive so it does not flip the value of the output

        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        # self.action_bound = action_bound

        # First layer
        #self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0]) # from the DDPG paper
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # Second layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0]) # from the DDPG paper
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Third layer
        f3 = 0.003 # from the DDPG paper
        # mu is the representation of the policy, the actual action not the probability (deterministic)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions) 
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x)) # bound the outcome in the range [-1, 1]

        return x
    
    def save_checkpoint(self):
        print(' ... Saving Checkpoint ... ')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(' ... Loading Checkpoint ... ')
        self.load_state_dict(T.load(self.checkpoint_file))

class DDPGAgent(object):
    # def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, 
    #             n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300,
    #             bathc_size=64):
    # UPDATED BASED OF THE RL PAPER
    def __init__(self, alpha, beta, input_dims, tau, env, gamma, 
                 n_actions, max_size, layer1_size, layer2_size, batch_size):

        # self.agent = DDPGAgent(alpha=LR_ACTOR, beta=LR_CRITIC, 
        #                   input_dims=INPUT_DIMENSION, 
        #                   tau=TAU, env=None, gamma=GAMMA,
        #                   n_actions=1, max_size=BUFFER_SIZE, 
        #                   layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE,
        #                   batch_size=BATCH_SIZE)
        
        self.gamma = gamma # from RL paper: gamma the discount factor = 1
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions=n_actions, name='Actor')
        
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions=n_actions, name='TargetActor')
        
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                    n_actions=n_actions, name='Critic')
        
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                    n_actions=n_actions, name='TargetCritic')
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1) # this solves the problem of the moving target

    def choose_action(self, observation):
        # self.actor.eval() # this programming step is very critical (if you use bn) for the agent to learn
        # observation = T.tensor(observation, dtype=T.float).to(self.actor.device) # this might need to be changed for cpu
        # mu = self.actor(observation).to(self.actor.device)
        # mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        # self.actor.train()

        # to pass np value instead of a tensor, which is not acceptable in openAI environment,
        # we apply the below operations/functions on the return value
        # return mu_prime.cpu().detach().numpy()

        print("Original observation shape:", observation.shape)

        # Convert observation to a tensor and ensure it is on the correct device
        observation_tensor = T.tensor(observation, dtype=T.float).to(self.actor.device)
        print("Tensor observation shape after conversion:", observation_tensor.shape)

        # Generate the action using the actor network
        self.actor.eval()  # Set the network to evaluation mode
        mu = self.actor(observation_tensor)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train() # Set the network back to training mode
        # Convert the action tensor back to a numpy array, if necessary
        action = mu_prime.cpu().detach().numpy()

        return action

    
    def remember(self, state, action, reward, new_state, done):
        # this function to store state transitions
        # it is just an interface for our ReplayBuffer memory class
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def learn(self):
        # this is the meat of the problem, where the learning actually happens
        if self.memory.mem_cntr < self.batch_size: 
            # you don't want to learn if you haven't filled at least batch size of your memory buffer
            return

        # otherwise 
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # turn the np arrays to tensors
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        # send everything to eval mode
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions) # the new state
        critic_value = self.critic.forward(state, action)
        # the above line: what is your estimate of the values of the states and actions
        # we actually encountered in our subset of the replay buffer

        # calculate the targets
        target = []
        for j in range(self.batch_size):
            # if the episode is over, then the value of the resulting state is multiplied by zero
            # so you only take into account the reward from the current state
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1) # reshape

        # send the critic back into training mode
        # we already performed the evaluation now we want to calculate the values of batch normalization
        self.critic.train()
        self.critic.optimizer.zero_grad() # in pytorch, zero your gradients before you calculate loss
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # send the critic to evaluation mode
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)

        # send the actor to train mode to calculate actor loss
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss) # derived from the paper
        actor_loss.backward()
        self.actor.optimizer.step()

        # now, we are done learning
        # you can now update your parameters for target_actor and target_critic
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        # tau: a parameter that allows the update of the target network to gradually 
        # approach the evaluation networks. important for nice slow conversions
        # tau is a small number much much less than 1
        # Tau value is not specified in the RL paper
        if tau is None:
            tau = self.tau # to start with the same weight for all networks
        
        actor_params = self.actor.named_parameters() # named_parameters() is a predefined method in PyTorch
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # save the parameters to dictionaries
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        # iterate over dictionaries and copy parameters
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
            (1-tau)*target_critic_dict[name].clone()
        
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
            (1-tau)*target_actor_dict[name].clone()
        
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
