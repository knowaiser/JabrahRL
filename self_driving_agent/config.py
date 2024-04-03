import os 

# dqn action values
action_values = [-0.75, -0.5, -0.25, -0.15, -0.1, -0.05, 0,
                0.05, 0.1, 0.15, 0.25, 0.5, 0.75]
action_map = {i:x for i, x in enumerate(action_values)}

env_params = {
    'target_speed' :18 , #30
    'max_iter': 4000,
    'start_buffer': 10,
    'train_freq': 1,
    'action_freq': 4,
    'save_freq': 200,
    'start_ep': 0,
    'max_dist_from_waypoint': 20 #20
}

config = {
    'checkpoint_dir': 'checkpoints',
    'actor_checkpoint': 'actor_checkpoint.pth',
    'target_actor_checkpoint': 'target_actor_checkpoint.pth',
    'critic_checkpoint': 'critic_checkpoint.pth',
    'target_critic_checkpoint': 'target_critic_checkpoint.pth',
}