import functools
import os

import d4rl
import gym
import numpy as np
import torch
import tqdm

import wandb
from dataset import D4RL_dataset
from BDiffusion import EDA_policy
from utils import get_args, marginal_prob_std, pallaral_simple_eval_policy


def train_policy(args, score_model, data_loader, start_epoch=0):
    n_epochs = args.n_policy_epochs
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    # evaluation_inerval = 4
    evaluation_inerval = 2
    mean, std = pallaral_simple_eval_policy(functools.partial(score_model.select_actions_sfbc, sample_per_state=4, base="policy"),args.env,00)
    args.run.log({"eval/rew{}".format("sfbc4mean"): mean}, step=0)
    args.run.log({"eval/rew{}".format("sfbc4std"): std}, step=0)
    # mean, std = pallaral_simple_eval_policy(score_model.diffusion_policy.select_actions,args.env,00)
    # args.run.log({"eval/rew{}".format("mean"): mean}, step=0)
    # args.run.log({"eval/rew{}".format("std"): std}, step=0)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = data_loader.sample(args.policy_batchsize)
            loss2 = score_model.update_policy(data)
            avg_loss += loss2
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        args.run.log({"train/loss": avg_loss / num_items}, step=epoch+1)
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            # mean, std = pallaral_simple_eval_policy(score_model.diffusion_policy.select_actions,args.env,00)
            # args.run.log({"eval/rew{}".format("mean"): mean}, step=epoch+1)
            # args.run.log({"eval/rew{}".format("std"): std}, step=epoch+1)
            mean, std = pallaral_simple_eval_policy(functools.partial(score_model.select_actions_sfbc, sample_per_state=4, base="policy"),args.env,00)
            args.run.log({"eval/rew{}".format("sfbc4mean"): mean}, step=epoch+1)
            args.run.log({"eval/rew{}".format("sfbc4std"): std}, step=epoch+1)
            # args.run.log({"info/lr": score_model.BDiffusion_policy_optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch+1)
    torch.save(score_model.state_dict(), os.path.join("./EDA_policy_models", str(args.expid), "policy_ckpt{}.pth".format(epoch+1)))
    torch.save(score_model.q[0].state_dict(), os.path.join("./EDA_policy_models", str(args.expid), "critic_ckpt{}.pth".format(epoch+1)))
def policy(args):
    for dir in ["./EDA_policy_models"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./EDA_policy_models", str(args.expid))):
        os.makedirs(os.path.join("./EDA_policy_models", str(args.expid)))
    run = wandb.init(project="EDA_policy", name=str(args.expid))
    wandb.config.update(args)
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.run = run
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=20.0)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    
    score_model= EDA_policy(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, actor_load_path=args.actor_load_path, args=args).to(args.device)
    score_model.q[0].to(args.device)

    if args.critic_load_path is not None:
        print("loadind critic...")
        ckpt = torch.load(args.critic_load_path, map_location=args.device)
        score_model.q[0].load_state_dict(ckpt)
    else:
        assert False

    dataset = D4RL_dataset(args)
    
    # generate support action set
    score_model.diffusion_behavior.eval()
    allstates = dataset.states[:].cpu().numpy()
    actions = []
    for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // 256 + 1)):
        actions.append(score_model.diffusion_behavior.sample(states, sample_per_state=args.M, diffusion_steps=15))
    actions = np.concatenate(actions)
    dataset.fake_actions = torch.Tensor(actions.astype(np.float32)).to(args.device)
    score_model.diffusion_behavior.train()
    

    print("training policy")
    train_policy(args, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()

if __name__ == "__main__":
    args = get_args()
    policy(args)