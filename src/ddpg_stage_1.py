#!/usr/bin/env python
# Authors: Junior Costa de Jesus #

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
from environment_stage_1 import Env
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
import math
from collections import deque

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))
#---Functions to make network updates---#
 
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau)+ param.data*tau)

def hard_update(target,source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#---Ornstein-Uhlenbeck Noise for action---#

class ActionNoise:
    # Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim)*self.mu
        
    def reset(self):
        self.X = np.ones(self.action_dim)*self.mu
    
    def sample(self):
        dx = self.theta*(self.mu - self.X)
        dx = dx + self.sigma*np.random.randn(len(self.X))
        self.X = self.X + dx
        print('aqu2i' + str(self.X))
        return self.X

#---Critic--#

EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.fa1 = nn.Linear(action_dim, 256)
        self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fca1 = nn.Linear(512, 512)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        
        self.fca2 = nn.Linear(512, 1)
        self.fca2.weight.data.uniform_(-EPS, EPS)
        
    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs,xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs

#---Actor---#

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        
        self.fa1 = nn.Linear(state_dim, 512)
        self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fa2 = nn.Linear(512, 512)
        self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())
        
        self.fa3 = nn.Linear(512, action_dim)
        self.fa3.weight.data.uniform_(-EPS,EPS)
        
    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if state.shape == torch.Size([14]):
            action[0] = torch.sigmoid(action[0])*self.action_limit_v
            action[1] = torch.tanh(action[1])*self.action_limit_w
        else:
            action[:,0] = torch.sigmoid(action[:,0])*self.action_limit_v
            action[:,1] = torch.tanh(action[:,1])*self.action_limit_w
        return action

#---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        
        return s_array, a_array, r_array, new_s_array
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, new_s):
        transition = (s, a, r, new_s)
        self.len += 1 
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)

#---Where the train is made---#

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

class Trainer:
    
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        #print('w',self.action_limit_w)
        self.ram = ram
        #self.iter = 0 
        self.noise = ActionNoise(self.action_dim)
        
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        self.qvalue = Float32()
        
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        
    def get_exploitation_action(self,state):
        state = torch.from_numpy(state)
        action = self.target_actor.forward(state).detach()
        #print('actionploi', action)
        return action.data.numpy()
        
    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        #noise = self.noise.sample()
        #print('noisea', noise)
        #noise[0] = noise[0]*self.action_limit_v
        #noise[1] = noise[1]*self.action_limit_w
        #print('noise', noise)
        new_action = action.data.numpy() #+ noise
        #print('action_no', new_action)
        return new_action
    
    def optimizer(self):
        s_sample, a_sample, r_sample, new_s_sample = ram.sample(BATCH_SIZE)
        
        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        
        #-------------- optimize critic
        
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        # y_exp = r _ gamma*Q'(s', P'(s'))
        y_expected = r_sample + GAMMA*next_value
        # y_pred = Q(s,a)
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))
        #-------Publisher of Vs------
        self.qvalue = y_predicted.detach()
        self.pub_qvalue.publish(torch.max(self.qvalue))
        #print(self.qvalue, torch.max(self.qvalue))
        #----------------------------
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        #------------ optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1*torch.sum(self.critic.forward(s_sample, pred_a_sample))
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)
    
    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), dirPath +'/Models/real1/'+str(episode_count)+ '_actor.pt')
        torch.save(self.target_critic.state_dict(), dirPath + '/Models/real1/'+str(episode_count)+ '_critic.pt')
        print('****Models saved***')
        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(dirPath + '/Models/real1/'+str(episode)+ '_actor.pt'))
        self.critic.load_state_dict(torch.load(dirPath + '/Models/real1/'+str(episode)+ '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('***Models load***')

#---Run agent---#

is_training = True

if is_training:
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.05
else:
    exploration_rate = 0.05
    max_exploration_rate = 0.05
    min_exploration_rate = 0.05

exploration_decay_rate = 0.001

MAX_EPISODES = 10001
MAX_STEPS = 500
MAX_BUFFER = 100000
rewards_all_episodes = []

STATE_DIMENSION = 14
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 2 # rad/s

if is_training:
    var_v = ACTION_V_MAX*0.20
    var_w = ACTION_W_MAX*2*0.20
else:
    var_v = ACTION_V_MAX*0.10
    var_w = ACTION_W_MAX*0.10

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')
ram = MemoryBuffer(MAX_BUFFER)
trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram)
trainer.load_models(200)


if __name__ == '__main__':
    rospy.init_node('ddpg_stage_1')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()

    start_time = time.time()
    past_action = np.array([0.,0.])

    for ep in range(MAX_EPISODES):
        done = False
        state = env.reset()
        print('Episode: ' + str(ep))

        rewards_current_episode = 0
        for step in range(MAX_STEPS):
            #print(step)
            state = np.float32(state)
            #action = trainer.get_exploration_action(state)
            #print('actionA:', action)
            
            #action[0] = np.clip(np.random.normal(action[0], var_v), 0., ACTION_V_MAX)
            #action[1] = np.clip(np.random.normal(action[1], var_w), -ACTION_W_MAX, ACTION_W_MAX)
            #print('actionD', action)

            if is_training:
                #if ep%2 == 0:
                #    action = trainer.get_exploitation_action(state)
                #    #print('aa', action)
                #else:
                action = trainer.get_exploration_action(state)
                #    #print('aa',action)
                action[0] = np.clip(np.random.normal(action[0], var_v), 0., ACTION_V_MAX)
                action[1] = np.clip(np.random.normal(action[1], var_w), -ACTION_W_MAX, ACTION_W_MAX)
                #action[0] = np.clip(action[0], 0., ACTION_V_MAX)
                #action[1] = np.clip(action[1], -ACTION_W_MAX, ACTION_W_MAX)
            #print('af', action) 

            #exploration_rate_threshold = random.uniform(0., 1.)
            #if exploration_rate_threshold > exploration_rate:
            #    action = trainer.get_exploration_action(state)
            #else:
            #    action = np.array([np.random.uniform(0,0.22), np.random.uniform(-1,1)])

            if not is_training:
                #print('nor')
                action = trainer.get_exploitation_action(state)
            #print('state',state)
            #print('action',action)
            #print('ap',past_action)
            next_state, reward, done = env.step(action, past_action)
            print('action', action,'r',reward)
            past_action = action

            rewards_current_episode += reward
            next_state = np.float32(next_state)
            ram.add(state, action, reward, next_state)
            state = next_state
            
            #action = np.array([np.random.uniform(0.,0.15), np.random.uniform(-0.5, 0.5)])
            #print('r: ' + str(reward) + ' and done: ' + str(done))

            if ram.len >= 2*MAX_STEPS and is_training:
                var_v = max([var_v*0.99999, 0.10*ACTION_V_MAX])
                var_w = max([var_w*0.99999, 0.10*ACTION_W_MAX])
                trainer.optimizer()
            #if is_training:
            #    trainer.optimizer()

            if done or step == MAX_STEPS-1:
                print('reward per ep: ' + str(rewards_current_episode))
                print('explore_v: ' + str(var_v) + ' and explore_w: ' + str(var_w))
                rewards_all_episodes.append(rewards_current_episode)
                result = rewards_current_episode
                pub_result.publish(result)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                break
        exploration_rate = (min_exploration_rate +
                (max_exploration_rate - min_exploration_rate)* np.exp(-exploration_decay_rate*ep))
        gc.collect()
        #print('exp:', exploration_rate)
        if ep%50 == 0:
            trainer.save_models(ep)

print('Completed Training')
