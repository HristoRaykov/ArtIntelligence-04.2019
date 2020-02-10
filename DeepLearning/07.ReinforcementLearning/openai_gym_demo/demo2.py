import numpy as np

import gym

env = gym.make("FrozenLake-v0")

EPSILON = 0.9  # exploration / exploitation

NUM_EPOCHS = 10000  # also called "episodes", number of games to play
MAX_STEPS = 100  # FrozenLake can never finish if no limit

learning_rate = 0.8
gamma = 0.95

Q = np.zeros((env.observation_space.n, env.action_space.n))  # 16 x 4


def choose_action(state):
	action = 0
	if np.random.uniform(0, 1) < EPSILON:
		action = env.action_space.sample()  # exploration, if this part is remove we have ML algorithm not RL
	else:
		action = np.argmax(Q[state, :])  # exploitation, greedy
	return action


def learn(state_t, state_t_plus_1, reward, action):
	predict = Q[state_t, action]  # take action, Q[state_t, action] should be replaced with NN model, forward propagation
	target = reward + gamma * np.max(Q[state_t_plus_1, :])  # calculate reward, take inference from the NN, Loss function
	Q[state_t, action] = Q[state_t, action] + learning_rate * (target - predict)  # update, backward propagation


for epoch in range(NUM_EPOCHS):
	state = env.reset()
	step = 0
	while step < MAX_STEPS:
		# env.render()
		action = choose_action(state)
		new_state, reward, done, info = env.step(action)
		learn(state, new_state, reward, action)
		state = new_state
		step += 1
		if done:
			break
