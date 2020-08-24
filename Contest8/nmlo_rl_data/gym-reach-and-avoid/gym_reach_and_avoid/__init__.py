import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ReachAndAvoid-v0',
    entry_point='gym_reach_and_avoid.envs:ReachAndAvoidEnv',
    timestep_limit=500,
)
