import copy

import torch
import torch.nn as nn

from model import *

class BDiffusion_Behavior(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        if ("v2" not in args.env) and ("kitchen" not in args.env):
            self.diffusion_behavior = Toy_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args).to(args.device)
            self.diffusion_optimizer = torch.optim.Adam(self.diffusion_behavior.parameters(), lr=1e-4)
        else:
            self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args).to(args.device)
            self.diffusion_optimizer = torch.optim.Adam(self.diffusion_behavior.parameters(), lr=3e-4)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0

    def update_behavior(self, data):
        self.step += 1
        all_a = data['a']
        all_s = data['s']

        # Update diffusion behavior
        self.diffusion_behavior.train()


        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1. - 1e-3) + 1e-3  
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        episilon = self.diffusion_behavior(perturbed_x, random_t, all_s)
        self.energy = self.diffusion_behavior.energy.detach().mean().cpu().item()
        assert episilon.shape == z.shape
        loss = torch.mean(torch.sum((episilon - z)**2, dim=(1,)))
        self.loss =loss

        self.diffusion_optimizer.zero_grad()
        loss.backward()  
        self.diffusion_optimizer.step()

class EDA_policy(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, actor_load_path, args=None):
        super().__init__()
        if ("v2" not in args.env) and ("kitchen" not in args.env):
            # for 2d bandit toy
            self.diffusion_behavior = Toy_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=32, args=args).to(args.device)
        else:
            self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=32, args=args).to(args.device)
        print("loading actor...")
        ckpt = torch.load(actor_load_path, map_location=args.device)
        self.load_state_dict({k:v for k,v in ckpt.items() if "diffusion_behavior" in k}) # TODO check load

        self.diffusion_policy = copy.deepcopy(self.diffusion_behavior).to(args.device)
        self.diffusion_behavior.eval()
        self.diffusion_optimizer = torch.optim.Adam(self.diffusion_policy.parameters(), lr=5e-5)
        
        # Disable dropout for behavior model
        for module in self.diffusion_policy.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        for module in self.diffusion_behavior.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.device=args.device
        self.output_dim = output_dim
        self.step = 0
    
        self.q = []
        if args.critic_type == "IQL":
            self.q.append(IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
        elif args.critic_type == "CEP":
            self.q.append(CEP_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
    
    def update_policy(self, data):
        self.step += 1
        # all_a = data['a']
        if 'fake_a' in data:
            all_s = data['s']
            fake_a = data['fake_a']
            concat_s = all_s.unsqueeze(1).expand(-1, fake_a.shape[1], -1)
            energy = self.q[0].q0_target(fake_a , concat_s).detach().squeeze()  # bz, M
        else:
            # for bandit toy data
            concat_s = data['s'].reshape((256,16,2))
            fake_a = data['a'].reshape((256,16,2))
            energy = data['e'].reshape((256,16))
            
        # Update diffusion behavior
        self.diffusion_policy.train()
        logsoftmax = nn.LogSoftmax(dim=1)
        softmax = nn.Softmax(dim=1)
        
        x0_data_energy = energy * self.args.alpha
            
        # random_t = torch.rand((fake_a.shape[0], fake_a.shape[1]), device=s.device) * (1. - 1e-3) + 1e-3
        random_t = torch.rand((fake_a.shape[0], ), device=concat_s.device) * (1. - 1e-3) + 1e-3
        random_t = random_t.unsqueeze(1).expand(-1, fake_a.shape[1])
        z = torch.randn_like(fake_a)
        alpha_t, std = self.marginal_prob_std(random_t)
        perturbed_fake_a = fake_a * alpha_t[..., None] + z * std[..., None]
        
        with torch.no_grad():
            baseline = self.diffusion_behavior.get_energy(perturbed_fake_a, random_t, concat_s)
            
            xt_model_energy = self.args.beta * (self.diffusion_policy.get_energy(perturbed_fake_a, random_t, concat_s) - baseline)
        p_label = softmax(x0_data_energy)
        # self.debug_used = torch.flatten(p_label).detach().cpu().numpy()
        loss = -torch.mean(torch.sum(p_label * logsoftmax(xt_model_energy), axis=-1))  #  <bz,M>

        self.diffusion_optimizer.zero_grad()
        loss.backward()  
        self.diffusion_optimizer.step()
        
        self.loss =loss.detach().mean().cpu().item()
        return self.loss

    def dpm_wrapper_sample(self, dim, batch_size, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()

    def select_actions_sfbc(self, states, diffusion_steps=15, sample_per_state=16, base="behavior"):
        multiple_input=True
        self.eval()# not here firstly when we evaluate all
        with torch.no_grad():
            states = states.to(self.diffusion_behavior.device)
            
            self.diffusion_behavior.condition = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.diffusion_policy.condition = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = torch.repeat_interleave(states, sample_per_state, dim=0)
            
            num_states = states.shape[0]
            # self.condition = states
            init_x = torch.randn(num_states*sample_per_state, self.output_dim, device=self.diffusion_behavior.device)
            if base == "behavior":
                results = self.diffusion_behavior.dpm_solver.sample(init_x, steps=diffusion_steps, order=2)
            elif base == "policy":
                results = self.diffusion_policy.dpm_solver.sample(init_x, steps=diffusion_steps, order=2)
            else:
                assert False
            actions = results.reshape(num_states, sample_per_state, self.diffusion_behavior.output_dim) # <bz, A>
            self.diffusion_behavior.condition = None
            self.diffusion_policy.condition = None
            self.condition = None
            
            rewards = self.q[0].q0_target(actions, states.unsqueeze(1).expand(-1, sample_per_state, -1))[...,0]
            max_indices = torch.argmax(rewards, dim=1)
            actions = torch.gather(actions, 1, max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, actions.size(-1))).squeeze(1)

        out_actions = actions.detach().cpu().numpy() if multiple_input else actions[0]
        self.train()
        return out_actions
        
class IQL_policy(nn.Module):
    def __init__(self, input_dim, output_dim, args=None):
        super().__init__()
        self.deter_policy = Dirac_Policy(output_dim, input_dim-output_dim).to("cuda")
        self.deter_policy_optimizer = torch.optim.Adam(self.deter_policy.parameters(), lr=3e-4)
        self.deter_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.deter_policy_optimizer, T_max=1500000, eta_min=0.)

        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.q.append(IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
    
    def update_iql(self, data):
        a = data['a']
        s = data['s']
        self.q[0].update_q0(data)
        
        # evaluate iql policy part, only for evaluation and debugging, can be deleted
        with torch.no_grad():
            target_q = self.q[0].q0_target(a, s).detach()
            v = self.q[0].vf(s).detach()
        adv = target_q - v
        temp = 10.0 if "maze" in self.args.env else 3.0
        if "kitchen" in self.args.env:
            temp = 0.5
        exp_adv = torch.exp(temp * adv.detach()).clamp(max=100.0)

        policy_out = self.deter_policy(s)
        bc_losses = torch.sum((policy_out - a)**2, dim=1)
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)
        self.deter_policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.deter_policy_optimizer.step()
        self.deter_policy_lr_scheduler.step()
        self.policy_loss = policy_loss
    
    def update_policy(self, data):
        a = data['a']
        s = data['s']        
        # evaluate iql policy part, can be deleted
        with torch.no_grad():
            target_q = self.q[0].q0_target(a, s).detach()
            v = self.q[0].vf(s).detach()
        adv = target_q - v
        temp = 10.0 if "maze" in self.args.env else 3.0
        if "kitchen" in self.args.env:
            temp = 0.5
        exp_adv = torch.exp(temp * adv.detach()).clamp(max=100.0)

        policy_out = self.deter_policy(s)
        bc_losses = torch.sum((policy_out - a)**2, dim=1)
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)
        self.deter_policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.deter_policy_optimizer.step()
        self.deter_policy_lr_scheduler.step()
        self.policy_loss = policy_loss

def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class IQL_Critic(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        self.q0 = TwinQ(adim, sdim, layers=args.q_layer).to(args.device)
        print(args.q_layer)
        self.q0_target = copy.deepcopy(self.q0).to(args.device)

        self.vf = ValueFunction(sdim).to("cuda")
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)
        self.discount = 0.99
        self.args = args
        self.tau = 0.9 if "maze" in args.env else 0.7
        print(self.tau)

    def update_q0(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        # Update Q function
        targets = r + (1. - d.float()) * self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        self.v = v.mean()
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self.v_loss = v_loss
        self.q_loss = q_loss
        self.q = target_q.mean()
        self.v = next_v.mean()
        # Update target
        update_target(self.q0, self.q0_target, 0.005)

class CEP_Critic(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        self.q0 = TwinQ(adim, sdim, layers=args.q_layer).to(args.device)
        print(args.q_layer)
        self.q0_target = copy.deepcopy(self.q0).to(args.device)

        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.discount = 0.99
        self.args = args

    def update_q0(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]

        fake_a_ = data['fake_a_']
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            next_energy = self.q0_target(fake_a_ , torch.stack([s_]*fake_a_.shape[1] ,axis=1)).detach().squeeze() # <bz, 16>   
            q_alpha = 20.0 if "maze" in self.args.env else 1.0         
            next_v = torch.sum(softmax(q_alpha * next_energy) * next_energy, dim=-1, keepdim=True)

        # Update Q function
        targets = r + (1. - d.float()) * self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self.q_loss = q_loss.detach().cpu().item()
        self.q_v = next_v.mean().detach().cpu().item()
        # Update target
        update_target(self.q0, self.q0_target, 0.005)
        return self.q_loss
