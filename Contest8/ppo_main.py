# Proximal Policy Optimization - ReachAndAvoid-v0
# Jason Chen & Srikar Gouru
# NMLO team name: GMD
# 17 August, 2020

"""
Resources: 
Explaination video of PPO: https://www.youtube.com/watch?v=5P7I-xPq8u8
Explaination article: https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
Actor/Critic PPO: https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6
Clean PyTorch implementation: https://github.com/JayChen35/RL-Adventure-2/blob/master/3.ppo.ipynb
Getting a single file from Github: $ curl -L -O https://github.com/JayChen35/RL-Adventure-2/raw/master/3.ppo.ipynb
Understanding PPO hyperparameters: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
Kernels: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
"""

import tensorflow as tf
import tensorflow.keras.backend as kb
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
import numpy as np
import gym
import gym_reach_and_avoid
import constants
import os
from datetime import datetime
from typing import Tuple


class Actor(tf.keras.Model):
    def __init__(self, obs_space: int, act_space: int):
        super(Actor, self).__init__()
        self.observation_space = obs_space
        self.action_space = act_space
        # The Actor explores the environment and maps states to actions (called the "policy").
        self.layer_1 = Dense(216, input_shape=(self.observation_space,), kernel_initializer="he_uniform", activation="relu")
        self.layer_2 = Dense(92, kernel_initializer="he_uniform", activation="relu")
        self.output_layer = Dense(self.action_space, kernel_initializer="glorot_uniform", activation="softplus")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=constants.ACTOR_LR)
        # The standard deviations in the action probability distribution.
        self.std = constants.STD_INITIAL
        self.sigma = tf.ones([1, self.action_space], dtype=tf.float64) * self.std
        # self.update_std = False  # This is set to true at the end of every episode
        self.call_count = 0

    def call(self, x: np.ndarray or tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.layer_1(x)
        x = self.layer_2(x)
        mu = self.output_layer(x)  # The means of the Gaussian distribution is the output of the Actor network
        if self.std > constants.STD_MINIMUM and self.call_count % 2000 == 0:
            self.std = self.std * constants.STD_DECAY
            self.sigma = tf.ones([1, self.action_space], dtype=tf.float64) * self.std
            # self.update_std = False
        self.call_count += 1
        return mu, self.sigma


class Critic(tf.keras.Model):
    def __init__(self, obs_space: int, act_space: int):
        super(Critic, self).__init__()
        self.observation_space = obs_space
        self.action_space = act_space
        # The Critic evalutates the Actor's policy and runs stochastic gradient descent on states->rewards.
        self.layer_1 = Dense(216, input_shape=(self.observation_space,), kernel_initializer="he_uniform", activation="relu")
        self.layer_2 = Dense(92, kernel_initializer="he_uniform", activation="relu")
        self.output_layer = Dense(1, kernel_initializer="glorot_uniform", activation=None)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=constants.CRITIC_LR)
        # The output of the Critic is a single value (the output of the Value Function).

    def call(self, x: np.ndarray or tf.Tensor) -> tf.float64:
        x = self.layer_1(x)
        x = self.layer_2(x)
        return tf.squeeze(self.output_layer(x))


class Trajectory:
    def __init__(self, state: np.ndarray, action: np.ndarray, log_prob: tf.Tensor, reward: float, advantage: tf.Tensor):
        super().__init__()
        # As defined in the paper, one "snapshot" of the environment and policy parameters is called a "Trajectory"
        self.state = state
        self.action = action
        self.log_prob = log_prob
        self.reward = reward
        self.advantage = advantage

    def to_tuple(self):
        return self.state, self.action, self.log_prob, self.reward, self.advantage


@tf.function
def train_step(actor: Actor, critic: Critic, trajectory: Trajectory):
    # Perform SGD on both the Actor and Critic with the PPO custom loss function
    with tf.GradientTape() as tape:
        tape.watch(actor.layer_1.variables)
        tape.watch(actor.layer_2.variables)
        tape.watch(actor.output_layer.variables)
        loss, entropy = ppo_loss(actor, critic, trajectory)
    gradients = tape.gradient(loss, actor.trainable_variables)
    actor.optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    # Stochastic gradient descent on the Critic
    with tf.GradientTape() as tape:
        tape.watch(critic.layer_1.variables)
        tape.watch(critic.layer_2.variables)
        tape.watch(critic.output_layer.variables)
        loss, entropy = ppo_loss(actor, critic, trajectory)
    gradients = tape.gradient(loss, critic.trainable_variables)
    critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    return loss, entropy  # Don't need to return loss, but returning to keep track of statistics


def run_actor_critic(actor: Actor, critic: Critic, state: np.ndarray) -> Tuple[tfp.distributions.TruncatedNormal, float]:
    # Simply a wrapper to return the intended outputs for the Actor and Critic (Actor can't return a Distribution)
    mu, sigma = actor(state)
    dist = tfp.distributions.TruncatedNormal(mu, sigma, constants.ACT_LOW_BOUND, constants.ACT_HIGH_BOUND)
    value_estimate = critic(state)
    return dist, value_estimate


def compute_gae(next_value: tf.Tensor, rewards: list, masks: list, values: list) -> list:
    # GAE = Generalized Advantage Estimation, the "real" return (first term in the advantage function)
    values.append(next_value)
    gae_value = 0
    returns = []
    # Mask (inverse of terminal) is used to prevent us from using a state from the next newly restarted game
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + constants.GAMMA * values[step+1] * masks[step] - values[step]
        gae_value = delta + constants.GAMMA * constants.GAE_LAMBDA * masks[step] * gae_value
        # GAE_lambda is a smoothing parameter to reduce variance in training
        returns.insert(0, gae_value + values[step])
    return returns


def ppo_loss(actor: Actor, critic: Critic, trajectory: Trajectory):
    state, action, old_log_probs, actual_return, advantage = trajectory.to_tuple()
    # Actual_return = the computed GAE
    dist, value = run_actor_critic(actor, critic, state)
    entropy = tf.reduce_mean(dist.entropy())  # reduce_mean() is just the mean
    new_log_probs = dist.log_prob(action)
    # Policy_ratio = r(theta) in the paper, basically how much more likely are we to take an action with this new policy?
    policy_ratio = tf.math.exp(new_log_probs-old_log_probs)  # Expressed as a difference of logs
    surr_1 = policy_ratio * advantage
    surr_2 = tf.clip_by_value(policy_ratio, 1.0-constants.CLIP_EPSILON, 1.0+constants.CLIP_EPSILON) * advantage
    # Actor loss follows the clipped objective, while Critic loss is just MSE between returns and the Critic prediction
    actor_loss = -1 * tf.reduce_mean(tf.minimum(surr_1, surr_2))
    critic_loss = tf.reduce_mean(tf.pow(actual_return-value, 2))
    # entropy = tf.reduce_mean(-(new_log_probs * (new_log_probs + 1e-10)))
    total_loss = constants.CRITIC_DISCOUNT * critic_loss + actor_loss - constants.ENTROPY_BETA * entropy
    return total_loss, entropy


def ppo_minibatch_iterator(states: list, actions: list, log_probs: list, returns: list, advantages: list):
    # Returns a Generator object (volatile Iterator) of size MINI_BATCH_SIZE to update pi (the policy)
    assert len(states) == constants.PPO_STEPS
    for _ in range(constants.PPO_STEPS // constants.MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, constants.PPO_STEPS, constants.MINI_BATCH_SIZE)
        for i in rand_ids:
            trajectory = Trajectory(states[i], actions[i], log_probs[i], returns[i], advantages[i])
            yield trajectory
            

def ppo_update(actor: Actor, critic: Critic, states: list, actions: list, log_probs: list, returns: list, advantages: list):
    loss_totals = []
    entropies = []
    # Epochs = the number of times we go through the ENTIRE batch of trajectories
    for epoch in range(constants.PPO_EPOCHS):
        # Grabs random mini-batches of trajectories several times until all the data is covered
        for trajectory in ppo_minibatch_iterator(states, actions, log_probs, returns, advantages):
            total_loss, entropy = train_step(actor, critic, trajectory)
            loss_totals.append(total_loss)
            entropies.append(entropy)
    return np.mean(loss_totals), np.mean(entropies)


def eval_current_policy(actor: Actor):
    state = np.reshape(env.reset(), [1, actor.observation_space])
    done = False
    total_reward = 0
    step = 0
    while not done and step < constants.PPO_STEPS:
        mu, sigma = actor(state)
        dist = tfp.distributions.TruncatedNormal(mu, sigma, constants.ACT_LOW_BOUND, constants.ACT_HIGH_BOUND)
        next_state, reward, done, _  = env.step(dist.sample()[0])
        next_state = np.reshape(next_state, [1, observation_space])
        total_reward += reward
        state = next_state
        step += 1
    return total_reward


def form_results():
    folder_name = "./{0}@{1}_ReachAndAvoid-PPO". \
        format(datetime.today().strftime('%Y-%m-%d'), datetime.now().time().strftime('%H%M'))
    tensorboard_path = folder_name + '/Tensorboard'
    saved_model_path = folder_name + '/Saved_models'
    checkpoint_path = folder_name + '/Checkpoints'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(tensorboard_path)
        os.makedirs(saved_model_path)
        os.makedirs(checkpoint_path)
    return tensorboard_path, saved_model_path, checkpoint_path


if __name__ == "__main__":
    kb.set_floatx("float64")  # Set float data type standard
    tf.config.run_functions_eagerly(True)  # Running eagerly, prevents @tf.function error
    env = gym.make(constants.ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    actor, critic = Actor(observation_space, action_space), Critic(observation_space, action_space)

    if constants.SAVE_MODEL:
        tensorboard_path, saved_model_path, checkpoint_path = form_results()

    test_rewards = []
    early_stop = False
    episode = 0
    while episode < constants.NUM_EPISODES and not early_stop:
        states    = []
        actions   = []
        log_probs = []
        rewards   = []
        masks     = []
        values    = []
        state = np.reshape(env.reset(), [1, observation_space])  # To feed into the network
        total_reward = 0
        for step in range(constants.PPO_STEPS):
            # This loop collects experiences gathered by the agent (Trajectories)
            if constants.RENDER:
                env.render()
            dist, value = run_actor_critic(actor, critic, state)
            action = dist.sample()[0]
            next_state, reward, terminal, _ = env.step(action)
            next_state = np.reshape(next_state, [1, observation_space])
            total_reward += reward
            log_prob = dist.log_prob(action)  # tf.Tensor

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(0 if terminal else 1)  # Append the opposite of terminal
            states.append(state)
            actions.append(action)
            state = next_state
            if terminal:
                state = np.reshape(env.reset(), [1, observation_space])  # Reset the environment

        next_value = critic(next_state)
        returns = compute_gae(next_value, rewards, masks, values)
        advantages = [tf.math.subtract(returns[x], values[x]) for x in range(len(returns))]
        advantages = tf.linalg.normalize(advantages)[0]
        avg_total_loss, avg_entropy = ppo_update(actor, critic, states, actions, log_probs, returns, advantages)
        print("[Episode {}] Total reward: {}. Average total loss: {}. Average entropy: {}. STD: {}.".format(
            episode, total_reward, avg_total_loss, avg_entropy, actor.std))
        episode += 1
        # actor.update_std = True  # Decay the STD every episode
        # Quickly test if the network has reached the threshold reward
        if constants.EVAL_WHILE_TRAIN and episode % constants.EVAL_EPISODE_INTERVAL == 0:
            test_reward = np.mean([eval_current_policy(actor) for _ in range(5)])
            test_rewards.append(test_reward)
            if test_reward >= constants.REWARD_THRESHOLD: early_stop = True
            if test_reward > max(test_rewards) and constants.SAVE_MODEL:
                actor.save(saved_model_path + '/Actor_eval', overwrite=True)
                critic.save(saved_model_path + '/Critic_eval', overwrite=True)
                print("Models saved on Episode {} since reward > previous average reward ({}>{}).".format(
                    episode, test_reward, max(test_rewards)))
            else:
                print("Models NOT saved since the current reward ({}) is not greater than the max reward ({}).".format(
                    test_reward, max(test_rewards)))
        # Save the model
        if constants.SAVE_MODEL and episode % constants.SAVE_INTERVAL == 0:
            actor.save(saved_model_path + '/Actor_interval', overwrite=True)
            critic.save(saved_model_path + '/Critic_interval', overwrite=True)
            print("Actor and Critic models successfully saved on Episode {}.".format(episode))
