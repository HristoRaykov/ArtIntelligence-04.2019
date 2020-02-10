import time
import gym

from agent import RandomAgent

env = gym.make("CartPole-v1")

agent = RandomAgent(env.action_space)

episode_count = 10
reward = 0
done = False

for i in range(episode_count):
	ob = env.reset()
	while True:
		action = agent.act(ob, reward, done)
		ob, reward, done, info = env.step(action)
		if done:
			print("Game Finished!")
			break
		env.render()
		time.sleep(1 / 30)
	env.close()
