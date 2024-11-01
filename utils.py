import argparse
import d4rl
import gym
import numpy as np
import torch


def marginal_prob_std(t, device="cuda",beta_1=20.0,beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """    
    t = torch.tensor(t, device=device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std

def simple_eval_policy(policy_fn, env_name, seed, eval_episodes=20):
    env = gym.make(env_name)
    env.seed(seed+561)
    all_rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()
        total_reward = 0.
        done = False
        while not done:
            with torch.no_grad():
                action = policy_fn(torch.Tensor(obs).unsqueeze(0).to("cuda")).cpu().numpy().squeeze()
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
            else:
                obs = next_obs
        all_rewards.append(d4rl.get_normalized_score(env_name, total_reward))
    return np.mean(all_rewards), np.std(all_rewards)

def pallaral_simple_eval_policy(policy_fn, env_name, seed, eval_episodes=20):
    eval_envs = []
    for i in range(eval_episodes):
        env = gym.make(env_name)
        eval_envs.append(env)
        env.seed(seed + 1001 + i)
        env.buffer_state = env.reset()
        env.buffer_return = 0.0
    ori_eval_envs = [env for env in eval_envs]
    import time
    t = time.time()
    while len(eval_envs) > 0:
        new_eval_envs = []
        states = np.stack([env.buffer_state for env in eval_envs])
        states = torch.Tensor(states).to("cuda")
        with torch.no_grad():
            actions = policy_fn(states)
        for i, env in enumerate(eval_envs):
            state, reward, done, info = env.step(actions[i])
            env.buffer_return += reward
            env.buffer_state = state
            if not done:
                new_eval_envs.append(env)
        eval_envs = new_eval_envs
    for i in range(eval_episodes):
        ori_eval_envs[i].buffer_return = d4rl.get_normalized_score(env_name, ori_eval_envs[i].buffer_return)
    mean = np.mean([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
    std = np.std([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
    print("rew: ")
    print(mean)
    print("std: ")
    print(std)

    return mean, std

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str)    
    parser.add_argument("--device", default="cuda", type=str)      
    parser.add_argument("--save_model", default=1, type=int)       
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1.0)  # default value can be found in Appendix E of the paper.
    parser.add_argument('--actor_load_path', type=str, default=None)
    parser.add_argument('--critic_load_path', type=str, default=None)
    parser.add_argument('--policy_batchsize', type=int, default=256)              
    parser.add_argument('--actor_blocks', type=int, default=3)     
    parser.add_argument('--q_layer', type=int, default=2)
    parser.add_argument('--n_policy_epochs', type=int, default=20) # set to 2 would be enough for most tasks
    parser.add_argument('--policy_layer', type=int, default=None)
    parser.add_argument('--critic_load_epochs', type=int, default=150)
    parser.add_argument('--alpha', type=float, default=3.0) # Linearly scale the Q-values, directly adapted from the CEP implementation
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--M', type=int, default=16)
    parser.add_argument('--model_type', type=str, default="MLPResNet")
    parser.add_argument('--critic_type', type=str, default="IQL")
    print("**************************")
    args = parser.parse_known_args()[0]
    if args.debug:
        args.actor_epoch =1
        args.critic_epoch =1
    print(args)
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
