BUFFER_SIZE = int(3e7)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 1            # discount factor
TAU = 1e-3              # for soft update of target parameters (not specified in the RL paper)
LR_ACTOR = 1e-5         # learning rate of the actor (alpha)
LR_CRITIC = 1e-5        # learning rate of the critic (beta)
WEIGHT_DECAY = 0        # L2 weight decay
LAYER1_SIZE = 1000      # Layer 1 of the critic\actor
LAYER2_SIZE = 1000      # Layer 2 of the critic\actor

# Environment related constants
INPUT_DIMENSION = 27


#DDPG Optimization (hyper)parameters
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 2e6
ACTION_STD_INIT = 0.2
TEST_TIMESTEPS = 5e4
DDPG_CHECKPOINT_DIR = 'preTrained_models/ddpg/'
POLICY_CLIP = 0.2

