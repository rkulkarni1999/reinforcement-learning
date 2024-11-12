#!/bin/bash

# Stops script if any script fails
set -e  

# # Project 1
nosetests -v Project1/mdp_dp_test.py

# # Project 2 - Part 1
nosetests -v Project2/Project2-1/mc_test.py

# Project 2 - Part 2 
nosetests -v Project2/Project2-2/td_test.py
