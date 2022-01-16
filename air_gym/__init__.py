import logging
from gym.envs.registration import register

#logger = logging.getLogger(__name__)

register(
    id='car_env-v0',
    entry_point='air_gym.envs:car_env',
    #timestep_limit=1000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)
