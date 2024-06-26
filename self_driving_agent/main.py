
from utils import *

import os
from environment import SimEnv
from DDPG.ddpg_torch import DDPGAgent
from config import env_params, action_map
from DDPG_parameters import *
from settings import *

def run():
    try:
        
        state_dim = INPUT_DIMENSION
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        num_actions = 1 # a single continuous action
        episodes = 10000

        #replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        #model = DQN(num_actions, state_dim, in_channels, device)

        # this only works if you have a model in your weights folder. Replace this by that file
        #model.load('weights/model_ep_4400')

        # set to True if you want to run with pygame
        env = SimEnv(visuals=False, **env_params)

        # Initialize DDPG agent
        agent = DDPGAgent(alpha=LR_ACTOR, beta=LR_CRITIC, 
                          input_dims=INPUT_DIMENSION, 
                          tau=TAU, env=None, gamma=GAMMA,
                          n_actions=1, max_size=BUFFER_SIZE, 
                          layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE,
                          batch_size=BATCH_SIZE)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(ep, eval=True)
            env.reset()
    finally:
        env.reset()
        env.quit()

if __name__ == "__main__":
    run()
