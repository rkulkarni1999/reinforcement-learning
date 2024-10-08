import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################

    def val_eval(s, depth, terminal = False):
        # # Termination condition for the tree depth. I am not doing recusive thing as it is super slow!!!!
        # if depth>5 or terminal == True:
        #     return 0.0
        
        val = 0
        for aindex in range(nA):
            probAgivenS = policy[s][aindex]
            # Access the chance of transitioning to the next states. 
            # we dont care about the ordering
            tups = P[sindex][aindex] 
            vThisIteration = 0
            for nextIndex in range(len(tups)):
                tup = tups[nextIndex]
                prob = tup[0]
                nextState = tup[1]
                reward = tup[2]
                terminal = tup[3]
                # print(sindex, " ", aindex, " reached, ", nextState, " ",  reward, " ", terminal)
                # nextEval = val_eval(nextState, depth+1, terminal) // This is super inefficient lol
                oldValueFuncForNextState = value_function[nextState]
                vThisIteration = vThisIteration + prob*(reward + gamma*oldValueFuncForNextState)

            val = val + probAgivenS*vThisIteration

        return val
                
    
    while True:
        delta = 0
        # Find the max delta among all the states for sucessive evaluations
        for sindex in range(nS):
            old_v = value_function[sindex]
            new_v = val_eval(sindex, 0)

            value_function[sindex] = new_v
            delta = max(delta, abs(new_v - old_v))

        if delta<tol:
            break

    # print(value_function)
    
    return value_function 


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """


    new_policy = np.ones([nS, nA]) / nA # policy as a uniform distribution
	############################
	# YOUR IMPLEMENTATION HERE #
    #                          #
	############################

    isPolStable = True

    for sindex in range(nS):
        oldAction = new_policy[sindex].copy()

        bestAction = 0
        bestActionReward = -100.0

        for aindex in range(nA):
            # Access the chance of transitioning to the next states. 
            # we dont care about the ordering
            tups = P[sindex][aindex] 
            vThisIteration = 0
            for nextIndex in range(len(tups)):
                tup = tups[nextIndex]
                prob = tup[0]
                nextState = tup[1]
                reward = tup[2]
                terminal = tup[3]
                oldValueFuncForNextState = value_from_policy[nextState]
                vThisIteration = vThisIteration + prob*(reward + gamma*oldValueFuncForNextState)
            if vThisIteration>bestActionReward:
                bestAction = aindex
                bestActionReward = vThisIteration

        newAction = np.zeros(4)
        newAction[bestAction] = 1 # Full probability to the best action

        new_policy[sindex] = newAction

    return new_policy



def policy_evaluation_v2(P, nS, nA, policy, oldValue, gamma=0.9, tol=1e-8):
    # Initialize value function as the old value (allows incremental updates)
    value_function = oldValue.copy()

    while True:
        delta = 0  # Track maximum change in value function
        
        # Update each state value based on the policy
        for s in range(nS):
            v = 0  # Temporary value for state s

            # Loop over all actions
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in P[s][a]:
                    v += action_prob * prob * (reward + gamma * value_function[next_state] * (not done))
            
            # Calculate maximum difference for convergence check
            delta = max(delta, abs(v - value_function[s]))
            value_function[s] = v

        # Break loop when change is below tolerance
        if delta < tol:
            break

    return value_function

def policy_improvement_v2(P, nS, nA, value_from_policy, old_policy, gamma=0.9):
    new_policy = np.zeros_like(old_policy)  # Initialize new policy

    isPolStable = True  # Flag to check if policy changes

    for s in range(nS):
        old_action = np.argmax(old_policy[s])  # Current action in the old policy
        best_action = None
        best_action_value = float('-inf')

        # Calculate the Q-value for each action
        for a in range(nA):
            action_value = 0
            for prob, next_state, reward, done in P[s][a]:
                action_value += prob * (reward + gamma * value_from_policy[next_state] * (not done))
            
            # Select the action with the highest Q-value
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = a

        # Update policy: assign 1 to the best action, 0 to others
        new_policy[s][best_action] = 1

        # Check if the policy has changed
        if old_action != best_action:
            isPolStable = False

    return new_policy, isPolStable

# def policy_improvement_v2(P, nS, nA, value_from_policy, old_policy, gamma=0.9):
#     # new_policy = np.ones([nS, nA]) / nA # policy as a uniform distribution
# 	############################
# 	# YOUR IMPLEMENTATION HERE #
#     #                          #
# 	############################

#     isPolStable = True

#     new_policy = old_policy.copy()
#     for sindex in range(nS):
#         oldAction = old_policy[sindex].copy()

#         bestAction = 0
#         bestActionReward = -100.0

#         for aindex in range(nA):
#             # Access the chance of transitioning to the next states. 
#             # we dont care about the ordering
#             tups = P[sindex][aindex] 
#             vThisIteration = 0
#             for nextIndex in range(len(tups)):
#                 tup = tups[nextIndex]
#                 prob = tup[0]
#                 nextState = tup[1]
#                 reward = tup[2]
#                 terminal = tup[3]
#                 oldValueFuncForNextState = value_from_policy[nextState]
#                 vThisIteration = vThisIteration + prob*(reward + gamma*oldValueFuncForNextState)
#             if vThisIteration>bestActionReward:
#                 bestAction = aindex
#                 bestActionReward = vThisIteration

#         newAction = np.zeros(nA)
#         newAction[bestAction] = 1 # Full probability to the best action

#         if np.array_equal(newAction, oldAction) == False:
#             isPolStable = False

#         new_policy[sindex] = newAction
#     return new_policy, isPolStable

def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    new_policy = policy.copy()
    V = np.zeros(nS)  # Initialize value function
    itr = 0

    while True:
        # Perform policy evaluation
        V = policy_evaluation_v2(P, nS, nA, new_policy, V, gamma, tol)

        # Perform policy improvement
        new_policy, policy_stable = policy_improvement_v2(P, nS, nA, V, new_policy, gamma)

        itr += 1

        if policy_stable:
            break  # If the policy is stable, we are done

    return new_policy.copy(), V.copy()


def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    value_function = V.copy()
    new_policy = np.zeros([nS, nA])

    def val_eval(s):
        val = -100000
        bestActionReward = -100
        bestAction = 0
        for aindex in range(nA):

            tups = P[sindex][aindex] 
            vCumsum = 0
            
            for nextIndex in range(len(tups)):
                tup = tups[nextIndex]
                prob = tup[0]
                nextState = tup[1]
                reward = tup[2]
                terminal = tup[3]
                oldValueFuncForNextState = value_function[nextState]
                vThisIteration = prob*(reward + gamma*oldValueFuncForNextState)
                vCumsum = vCumsum + vThisIteration

            val = max(val, vCumsum)

        return val
                
    
    while True:
        delta = 0
        # Find the max delta among all the states for sucessive evaluations
        for sindex in range(nS):
            old_v = value_function[sindex]
            new_v = val_eval(sindex)
            value_function[sindex] = new_v

            delta = max(delta, abs(new_v - old_v))
        
        if delta<tol:
            break

    new_policy = policy_improvement(P, nS, nA, value_function, gamma=0.9)

    return new_policy, value_function

def render_single(env, policy, render=False, n_episodes=100):
    """
    Given a game environment, play multiple episodes using the given policy.
    An episode ends when 'done' is True, which can happen either when the agent
    reaches the goal or falls into a hole.
    
    Parameters:
    ----------
    env: gym.core.Environment
        Environment to play in. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
        The action to take at a given state.
    render: bool
        Whether to render the game on each step (slows down the simulation).
    n_episodes: int
        Number of episodes to play.

    Returns:
    --------
    total_rewards: int
        Total accumulated rewards across all episodes.
    """
    total_rewards = 0

    for _ in range(n_episodes):
        ob, _ = env.reset()  # Reset the environment
        done = False
        
        while not done:
            if render:
                env.render()  # Render the game if render=True

            # Choose action based on the policy
            action = np.argmax(policy[ob])  # Or sample if using stochastic policy
            
            # Take the action in the environment
            ob, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Consider both done and truncated states

            # Accumulate the reward
            total_rewards += reward

            # Debugging output
            # print(f"State: {ob}, Action: {action}, Reward: {reward}, Done: {done}")

    return total_rewards




