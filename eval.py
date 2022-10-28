import ray
import gym
from env import CartPoleRightLeft

from ray.rllib.algorithms import dqn

save_dir = "videos/"
env = CartPoleRightLeft(env_config={})
env = gym.wrappers.RecordVideo(env=env, video_folder=save_dir)

ray.init()

config = dqn.DQNConfig()
config.training(
    num_atoms=2,
    v_min=0.,
    v_max=200.,
    noisy=True,
    #dueling=True,
    #double_q=True,
    hiddens=[256],
    n_step=5
).resources(num_gpus=1).framework(framework="torch")

model = dqn.DQN(env=CartPoleRightLeft,config=config)
model.restore("save/checkpoint_001940")

done    = False
obs     = env.reset()
env.goal= 1
policy  = model.get_policy()

while not done:
  action, _, _ = policy.compute_single_action(obs=obs)
  obs, _, done, _ = env.step(action=action)


