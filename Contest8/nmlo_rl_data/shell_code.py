# from: https://github.com/openai/gym/blob/master/examples/agents/random_agent.py 
import gym 
import gym_reach_and_avoid

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, observation, reward, done):
        return self.action_space.sample()

env = gym.make('ReachAndAvoid-v0')
env.seed(0)
agent = RandomAgent(env.action_space)

episode_count = 100
reward = 0
done = False

for i in range(episode_count):
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        # action = (.25, .1)
        ob, reward, done, _ = env.step(action)
        env.render()
        if done:
            break

env.close() 