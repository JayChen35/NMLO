import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math
import random
from math import sin, cos 

class ReachAndAvoidEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_ang_vel = 0.5 # rad/sec 
        self.dt = 0.1 # sec 
        self.single_arm_length = 10 # meters
        self.env_radius = 20 # meters
        self.viewer = None
        self.contact_radius_goal = 1 # meters 
        self.contact_radius_avoid = 8 # meters 

        self.action_space = spaces.Box(
            np.array([-1, -1]),
            np.array([1, 1]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            np.array([0, 0, -self.env_radius, -self.env_radius, -self.env_radius, -self.env_radius]),
            np.array([2 * math.pi, 2 * math.pi, self.env_radius, self.env_radius, self.env_radius, self.env_radius]),
            dtype = np.float32
        )

        self.seed()

    def seed(self, seed=None):
        random.seed()

    def step(self, action):
        arm1_rot, arm2_rot, goal_x, goal_y, avoid_x, avoid_y = self.state 

        action = np.clip(action, -self.max_ang_vel, self.max_ang_vel)

        arm1_ang_vel = action[0] * self.max_ang_vel 
        arm2_ang_vel = action[1] * self.max_ang_vel 

        arm1_rot += arm1_ang_vel * self.dt 
        arm2_rot += arm2_ang_vel * self.dt 

        arm1_rot %= 2 * math.pi 
        arm2_rot %= 2 * math.pi 

        arm1_end_pos = (self.single_arm_length * cos(arm1_rot), self.single_arm_length * sin(arm1_rot)) 
        arm2_end_pos = (arm1_end_pos[0] + self.single_arm_length * cos(arm1_rot + arm2_rot), arm1_end_pos[1] + self.single_arm_length * sin(arm1_rot + arm2_rot)) 

        dist_end_effector_goal = math.sqrt(pow(arm2_end_pos[0] - self.state[2], 2) + pow(arm2_end_pos[1] - self.state[3], 2)) 
        dist_end_effector_avoid = math.sqrt(pow(arm2_end_pos[0] - self.state[4], 2) + pow(arm2_end_pos[1] - self.state[5], 2)) 

        reward = -dist_end_effector_goal 

        done = False 

        if dist_end_effector_goal < self.contact_radius_goal: 
            reward += 100 
            done = True 

        if dist_end_effector_avoid < self.contact_radius_avoid: 
            reward -= 100 
            done = True

        self.state = np.array([arm1_rot, arm2_rot, goal_x, goal_y, avoid_x, avoid_y])
        return self._get_obs(), reward, done, {}

    def reset(self):
        goal_x = random.uniform(-self.env_radius, self.env_radius) 
        goal_y = random.uniform(-self.env_radius, self.env_radius) 
        avoid_x = random.uniform(-self.env_radius, self.env_radius) 
        avoid_y = random.uniform(-self.env_radius, self.env_radius) 
        while (math.sqrt(pow(goal_x, 2)+pow(goal_y, 2)) >= self.env_radius-self.contact_radius_goal) or (math.sqrt(pow(avoid_x, 2)+pow(avoid_y, 2)) >= self.env_radius-self.contact_radius_avoid) or (math.sqrt(pow(goal_x-avoid_x, 2)+pow(goal_y-avoid_y, 2)) < self.contact_radius_avoid+self.contact_radius_goal): 
            goal_x = random.uniform(-self.env_radius, self.env_radius) 
            goal_y = random.uniform(-self.env_radius, self.env_radius) 
            avoid_x = random.uniform(-self.env_radius, self.env_radius) 
            avoid_y = random.uniform(-self.env_radius, self.env_radius) 

        self.state = np.array([random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), goal_x, goal_y, avoid_x, avoid_y])
        return self._get_obs()

    def _get_obs(self):
        state = self.state
        return state 

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-22, 22, -22, 22)
            self.goal_transform = rendering.Transform()
            goal = rendering.make_circle(self.contact_radius_goal)
            goal.set_color(.1, .7, .1)
            goal.add_attr(self.goal_transform)
            self.viewer.add_geom(goal)
            self.avoid_transform = rendering.Transform()
            avoid = rendering.make_circle(self.contact_radius_avoid)
            avoid.set_color(.7, .1, .1)
            avoid.add_attr(self.avoid_transform)
            self.viewer.add_geom(avoid)
            arm1 = rendering.make_capsule(self.single_arm_length, 1)
            arm2 = rendering.make_capsule(self.single_arm_length, 1)
            arm1.set_color(.1, .4, .7)
            arm2.set_color(.4, .1, .7)
            self.arm1_transform = rendering.Transform()
            arm1.add_attr(self.arm1_transform)
            self.viewer.add_geom(arm1)
            self.arm2_transform = rendering.Transform()
            arm2.add_attr(self.arm2_transform)
            self.viewer.add_geom(arm2)
            axle = rendering.make_circle(0.6)
            axle.set_color(.4, .4, .4)
            self.viewer.add_geom(axle)
            self.end_transform = rendering.Transform()
            end_effector = rendering.make_circle(self.contact_radius_goal)
            end_effector.set_color(.7, .7, .7)
            end_effector.add_attr(self.end_transform)
            self.viewer.add_geom(end_effector)
            bounds = rendering.make_circle(self.env_radius + self.contact_radius_goal/2, filled=False)
            bounds.set_color(.4, .4, .4)
            self.viewer.add_geom(bounds)
                        
        self.arm1_transform.set_rotation(self.state[0])

        arm1_end_pos = (self.single_arm_length * cos(self.state[0]), self.single_arm_length * sin(self.state[0])) 

        self.arm2_transform.set_translation(arm1_end_pos[0], arm1_end_pos[1])
        self.arm2_transform.set_rotation(self.state[1]+self.state[0])

        arm2_end_pos = (arm1_end_pos[0] + (self.single_arm_length * cos(self.state[0] + self.state[1])), arm1_end_pos[1] + (self.single_arm_length * sin(self.state[0] + self.state[1])))         
        self.end_transform.set_translation(arm2_end_pos[0], arm2_end_pos[1])

        self.goal_transform.set_translation(self.state[2], self.state[3])

        self.avoid_transform.set_translation(self.state[4], self.state[5])
        
        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array = False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
