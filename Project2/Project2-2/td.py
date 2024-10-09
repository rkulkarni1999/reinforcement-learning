#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0

import numpy as np
import random
from collections import defaultdict

def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Select action using epsilon-greedy strategy."""
    return np.argmax(Q[state]) if random.random() > epsilon else random.randint(0, nA - 1)

def select_action_sarsa(Q, state, epsilon):
    """Select action using the SARSA policy."""
    return np.argmax(Q[state]) if random.random() > epsilon else random.randint(0, 3)


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for x in range(n_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        action = select_action_sarsa(Q, state, epsilon)
        
        while (not terminated) and (not truncated):
            state_, reward, terminated, truncated, info = env.step(action)
            action_ = select_action_sarsa(Q, state_, epsilon)
            Q[state][action] += alpha * (reward + gamma * Q[state_][action_] - Q[state][action])
            state, action = state_, action_
            
        epsilon = 0.99 * epsilon
    
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while (not terminated) and (not truncated):
            action = select_action_sarsa(Q, state, epsilon)
            state_, reward, terminated, truncated, info = env.step(action)
            action_ = select_action_sarsa(Q, state_, 0)
            Q[state][action] += alpha * (reward + gamma * Q[state_][action_] - Q[state][action])
            state = state_
            
        epsilon = 0.99 * epsilon
    
    return Q
