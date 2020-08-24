# Jason Chen & Srikar Gouru
# NMLO team name: GMD
# 17 August, 2020

import tensorflow as tf
import numpy as np
import gym
import gym_reach_and_avoid
from collections import deque
import time
import random

import constants

# Notes:
# Continuous action space from -1 to 1 which is scaled to the correct angular velocities
# Observation space: [rotation of arm base "shoulder" joint, rotation of arm second "elbow" joint, 
# x-position of the goal, y-position of the goal, x-position of the center of the avoid-area, 
# y-position of the avoid-area] (radians from 0 to 2pi, meters)

class DQNSolver:
    def __init__(self, len_observation_space: int, len_action_space: int):
        super().__init__()
        self.observation_space = len_observation_space
        self.action_space = len_action_space
        self.exploration_rate = constants.STARTING_EPSILON
        self.memory = deque(maxlen=int(constants.MEMORY_SIZE))
        
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(50, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(tf.keras.layers.Dense(25, activation="relu"))
        self.model.add(tf.keras.layers.Dense(self.action_space, activation=None))
        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=constants.LEARNING_RATE))

    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state: np.ndarray):
        # Choose a course of action and return the index of the max q-value
        if np.random.rand() < self.exploration_rate:
            return None  # Do random action
        q_values = self.model.predict(state)
        # return np.argmax(q_values[0])
        return q_values[0]

    def experience_replay(self):
        if len(self.memory) < constants.BATCH_SIZE:
            return
        batch = random.sample(self.memory, constants.BATCH_SIZE)
        for state, action, reward, next_state, terminal in batch:
            q_update = reward
            if not terminal:  # If it's not done, take into account possible future rewards with a discount
                q_update = (reward + constants.GAMMA * np.amax(self.model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update  # Error here since DQN can't work with a continuous action space!
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= constants.EPSILON_DECAY
        self.exploration_rate = max(constants.MIN_EPSILON, self.exploration_rate)

if __name__ == "__main__":
    env = gym.make(constants.ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    dqn_solver = DQNSolver(observation_space, action_space)
    for episode in range(constants.NUM_EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])  # To feed into the network
        terminal = False
        step = 1
        total_reward = 0
        while terminal == False and step <= constants.MAX_STEPS:
            if constants.RENDER:
                env.render()
            action = dqn_solver.select_action(state)  # Returns index of best action
            if action is None:
                action = env.action_space.sample()
            # print(action, action.shape)
            # example_action = env.action_space.sample()
            # print(example_action, example_action.shape)
            state_next, reward, temp_terminal, info = env.step(action)
            terminal = temp_terminal
            total_reward += reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            dqn_solver.experience_replay()
            if terminal:
                print("Episode {}: [Total reward = {}] [Total steps = {}] [Exploration = {}]".format(
                    episode, total_reward, step, dqn_solver.exploration_rate
                ))
            step += 1
