#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
    
class DQN(nn.Module):
    
    def __init__(self, in_channels=4, num_actions=4):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(20736, 100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, num_actions)
        )

    def forward(self, x):
        # Normalize input
        x = x / 255.0
        x = self.network(x)
        
        return x
