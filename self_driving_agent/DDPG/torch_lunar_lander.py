from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              bathc_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = [] # to keep score over time

for i in range(1000): # iterate over a thousand games
    done = False
    score = 0
    obs = env.reset() # new observations
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        # we learn on every step because this is a temporal difference learning method
        # as oppose to Monte Carlo where we would learn at the end of every episode
        agent.learn() 
        score += reward
        obs = new_state
    
    # print the place marker at the end of every episode
    score_history.append(score)
    # print the score 
    # and print the mean of the last 100 scores
    print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

    # save the model every 25 games
    if i % 25 == 0:
        agent.save_models()

filename = 'lunar-lander.png'
plotLearning(score_history, filename, window=100)




