#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    
    return 0 if observation[0] >= 20 else 1


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    
    # Initialize dictionaries to store returns and counts
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)  # Value function

    for _ in range(n_episodes):
        state, _ = env.reset()  # Start new episode
        episode = []  # Store state, action, reward tuples

        # Generate an episode
        while True:
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, reward))  # Track state and reward
            if terminated or truncated:
                break
            state = next_state

        # First-visit Monte Carlo update for each state
        visited = set()
        G = 0  # Return starts at the end of the episode
        for state, reward in reversed(episode):
            G = gamma * G + reward  # Compute return
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]  # Update value function

    return V



def epsilon_greedy(Q, state, nA, epsilon=0.1):
    
    if random.random() > epsilon:
        # Exploit: choose the best action with the highest Q-value
        return np.argmax(Q[state])
    else:
        # Explore: choose a random action
        return random.randint(0, nA - 1)


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    
    # Initialize Q-value table and returns tracking
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for i in range(n_episodes):
        state, _ = env.reset()
        episode = []
        visited_in_episode = defaultdict(lambda: defaultdict(bool))

        # Generate the episode
        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            if terminated or truncated:
                break
            state = next_state

        # First-visit Monte Carlo control
        G = 0  # Cumulative reward
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if not visited_in_episode[state][action]:
                visited_in_episode[state][action] = True
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]

        # Decay epsilon for better exploitation later in training
        epsilon = max(0.1, epsilon - 0.1 / n_episodes)

    return Q

