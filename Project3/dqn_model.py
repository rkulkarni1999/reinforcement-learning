#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

def dp(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)
    
class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """
    

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        out = 32
        multiplier = 2
        kernel_size = 5
        stride = 2
        self.network = nn.Sequential(
            
            
            nn.Conv2d(4, out, kernel_size, stride), 
            nn.BatchNorm2d(out), 
            nn.ReLU(), 
            
            nn.Conv2d(out, out*multiplier, kernel_size, stride),
            nn.BatchNorm2d(out*multiplier),
            nn.ReLU(),

            nn.Flatten(),

            nn.Dropout(0.25),
            nn.Linear(20736, 100),
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.Linear(100, 4),
            nn.ReLU(),
            
        )


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        
        x = x/255.0
        y = self.network(x)
        return y
        # return x
