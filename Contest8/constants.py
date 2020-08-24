# Environment
ENV_NAME = "ReachAndAvoid-v0"
AVOID_RADIUS = 8
GOAL_RADIUS = 1
ARM_SECTION_LENGTH = 10
TOTAL_ARM_LENGTH = 20
BOARD_RADIUS = 20
EPISODE_LENGTH = 50  # seconds
TIMESTEP = 0.1  # seconds
ACT_LOW_BOUND = -1  # Low bound of the action space
ACT_HIGH_BOUND = 1

NUM_EPISODES = 500
PPO_STEPS = 200 # 500  # Episode length/timestep
RENDER = False
EVAL_WHILE_TRAIN = False
EVAL_EPISODE_INTERVAL = 2
SAVE_MODEL = True
SAVE_INTERVAL = 10

# RL parameters
HAND_REWARD = 100
HAND_PENALTY = -100

# Hyperparameters
ACTOR_LR = 1e-04
CRITIC_LR = 1e-04
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE smoothing factor
MINI_BATCH_SIZE = 20
CLIP_EPSILON = 0.2  # Clippling parameter
PPO_EPOCHS = 4  # How much to train on a single batch of experience (PPO is on-policy)
REWARD_THRESHOLD = 90
CRITIC_DISCOUNT = 0.5  # 0.5  # c1 in the paper (Value Function Coefficient)
ENTROPY_BETA = 0.05 # 0.01  # c2 in the paper

STD_INITIAL = 0.5  # Wide initial standard deviation to help exploration
STD_DECAY = 0.9
STD_MINIMUM = 0.05
