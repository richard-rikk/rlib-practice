import ray
from env import CartPoleRightLeft

from ray.rllib.algorithms import dqn
from ray.tune.logger      import pretty_print

ray.init()

config = dqn.DQNConfig()
config.training(
    num_atoms=2,
    v_min=0.,
    v_max=200.,
    noisy=True,
    dueling=True,
    double_q=True,
    hiddens=[128,128],
    n_step=5
).resources(num_gpus=1).framework(framework="torch")

model = dqn.DQN(env=CartPoleRightLeft,config=config)

epoch                 = 2
max_mean_score        = 0.
episode_len_mean      = []
episode_reward_mean   = []
episode_reward_max    = []
mean_td_error         = []
save_dir              = "save/"

for i in range(epoch):
    result = model.train()
    # print(pretty_print(result=result))

    # Save metrics
    episode_len_mean.append(result["episode_len_mean"])
    episode_reward_mean.append(result["episode_reward_mean"])
    episode_reward_max.append(result["episode_reward_max"])
    mean_td_error.append(result["info"]["learner"]["default_policy"]["mean_td_error"])

    if episode_reward_mean[-1] > max_mean_score:
        max_mean_score = episode_reward_mean[-1]
        model.save(checkpoint_dir=save_dir)

    print(f"Epoch finished: {i+1} -- Mean reward: {episode_reward_mean[-1]}")

ray.shutdown()