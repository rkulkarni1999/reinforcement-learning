import json
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open('breakout_rl_local_00001.json', 'r') as f:
    data = json.load(f)

# Extract the number of episodes and the average rewards from the JSON data
episodes = [entry[1] for entry in data]  # Second item in each entry represents the episode number
average_rewards = [entry[2] for entry in data]  # Third item in each entry represents the average reward

# Plotting the learning curve
plt.figure(figsize=(10, 5))
plt.plot(episodes, average_rewards, label="Average Reward (last 30 episodes)")
plt.xlabel("Number of Episodes")
plt.ylabel("Average Reward")
plt.title("Learning Curve of DQN")
plt.legend()
plt.grid(True)
plt.show()
