import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import gym
import constants
import gym_reach_and_avoid

def eval_current_policy(actor, observation_space):
    state = np.reshape(env.reset(), [1, observation_space])
    done = False
    total_reward = 0
    step = 0
    while not done:
        if RENDER:
            env.render()
        mu, sigma = actor(state)
        dist = tfp.distributions.TruncatedNormal(mu, sigma, constants.ACT_LOW_BOUND, constants.ACT_HIGH_BOUND)
        next_state, reward, done, _  = env.step(dist.sample()[0])
        next_state = np.reshape(next_state, [1, observation_space])
        total_reward += reward
        state = next_state
        step += 1
    return total_reward

if __name__ == "__main__":
    RENDER = True
    path = "2020-08-23@0213_ReachAndAvoid-PPO/Saved_models/Actor_interval"
    env = gym.make(constants.ENV_NAME)
    observation_space = env.observation_space.shape[0]
    model = tf.keras.models.load_model(path)
    for ep in range(100):
        reward = eval_current_policy(model, observation_space) 
        print("Episode {} reward: {}.".format(ep, reward))
