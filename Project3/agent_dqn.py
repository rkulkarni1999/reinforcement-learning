#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing Modules
import random
import numpy as np
from collections import deque, namedtuple
import os
import sys
import math
import shutil
import time

from agent import Agent
from dqn_model import DQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Seeding for consistency of results
torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

# Output Directory
output_dir = "./outputs/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

class Agent_DQN(Agent):
    def __init__(self, env, args):
    
        super(Agent_DQN,self).__init__(env)
        
        # Defining Hyper-parameters
        self.epsilon_start = 1
        self.epsilon_end = 0.025
        self.epsilon_decay = 1000        
        self.clip_gradient = 1
        self.tau = 0.001
        self.gamma = 0.99
        
        # training params
        self.batch_size = 16 
        self.learning_rate = 0.0001
        self.num_episodes = 100000
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Defining the Network
        self.policy_net = DQN().to(device=self.device)
        self.target_net = DQN().to(device=self.device)
        
        # load state dicts
        self.policy_net.load_state_dict(torch.load("outputs_final/models_final/model_old.pth", map_location=torch.device(self.device)))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # define optimizers
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
            
        # replay buffer
        self.transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
        self.replay_buff = deque([], maxlen=10000)
        
        # define tensorboard writer for logging
        self.writer = SummaryWriter(log_dir=f"runs/breakout_rl_local")

        # Initializers
        self.current_episode = 0
        self.action_steps_done = 0
        self.step_counter = 0

        # Executes if Test is True
        if args.test_dqn:
            # Load the trained model
            self.policy_net.load_state_dict(torch.load("outputs_final/models_final/model_final.pth", map_location=self.device))    
            self.policy_net.eval()

    def init_game_setting(self):
        pass
        
    def make_action(self, observation, test=True):

        sample = random.random()
        self.epsilon_threshold = self.epsilon_start - (self.epsilon_start - self.epsilon_end) / np.float32(self.num_episodes) * np.float32(self.current_episode)
        self.action_steps_done += 1
        if test or sample > self.epsilon_threshold:
            with torch.no_grad():
                observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
                observation_tensor = observation_tensor.permute([2,0,1]).unsqueeze(0)
                outputs = self.policy_net(observation_tensor)
                action = torch.argmax(outputs).item()
        else:
            action = random.choice([0,1,2,3])
        return action
    
    def push(self, *args):
        self.replay_buff.append(self.transition(*args))
        
    def replay_buffer(self):
        return random.sample(self.replay_buff, self.batch_size)
    
    def train(self):
        
        reward_history = deque(maxlen=30)
        for ii in range(self.num_episodes):
            self.current_episode = ii
            state = self.env.reset()
            sum_reward = 0.0
            tIndex, sum_reward, episode_start_time = 0, 0.0, time.time()
            
            while True:
                
                action = self.make_action(state)
                observation, reward, done, truncated, _ = self.env.step(action)
                sum_reward += reward
                
                next_state = None if done or truncated else observation
                self.push(state, action, next_state, reward)
    
                state = next_state

                if len(self.replay_buff) < self.batch_size:
                    continue
                
                transitions = self.replay_buffer()
                batch = self.transition(*zip(*transitions))
                
                non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
                non_final_next_state_list = [torch.tensor(s, device=self.device).unsqueeze(0) for s in batch.next_state if s is not None]        
                non_final_next_state = torch.cat(non_final_next_state_list, dim=0)
            
                state_batch = torch.tensor(np.array(batch.state), device=self.device)
                action_batch = torch.tensor(batch.action, device=self.device)
                reward_batch = torch.tensor(batch.reward, device=self.device)
                state_batch = state_batch.permute([0,3,1,2])
                
                batch_policy = torch.zeros(self.batch_size, device=self.device)
                batch_policy_all = self.policy_net(state_batch)
                
                for idx in range(self.batch_size):
                    
                    batch_policy[idx] = batch_policy_all[idx][action_batch[idx]]
                    
                batch_target_next = torch.zeros(self.batch_size, device=self.device)

                with torch.no_grad():
                    non_final_next_state = non_final_next_state.permute([0,3,1,2])
                    batch_target_next[non_final_mask] = self.target_net(non_final_next_state).max(1)[0]
            
                expected_state_action_values = ( batch_target_next*self.gamma ) + reward_batch
                criterion = nn.SmoothL1Loss()
                loss = criterion(batch_policy, expected_state_action_values)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.clip_gradient)
                self.optimizer.step()
                                
                self.step_counter += 1
                if self.step_counter%100 == 0:
                    self.writer.add_scalar("Epsilon", self.epsilon_threshold, self.step_counter)
                                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                if done:
                    break
                
                tIndex += 1
            
            print(f"Episode Number: {ii}")
            
            episode_elapsed_time = time.time() - episode_start_time
            reward_history.append(sum_reward)
            
            # Calculate and log average reward over the last 30 episodes
            if len(reward_history) == 30:
                avg_reward_last_30 = sum(reward_history) / 30
                self.writer.add_scalar("Average Reward (Last 30 Episodes)", avg_reward_last_30, ii)

            if ii % 10 == 0:
                self.writer.add_scalar("Episode Reward", sum_reward, ii)
                self.writer.add_scalar("Episode Time", episode_elapsed_time, ii)
                self.writer.add_scalar("Episode Max Steps", tIndex, ii)

                   
            if ii%100 == 0:
                model_path = output_dir + f"model_{ii}.pth"
                torch.save(self.policy_net.state_dict(), model_path)
                
            