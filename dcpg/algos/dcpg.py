from collections import deque
import copy

import numpy as np

import torch
from torch.distributions import kl_divergence
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler


class Buffer:
    def __init__(self, buffer_size, device, store_unnormalised_obs=True):
        self.segs = deque(maxlen=buffer_size)
        self.store_unnormalised_obs = store_unnormalised_obs
        self.device = device
        self.indices_to_sample = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.segs)

    def insert(self, seg, step):
        if self.store_unnormalised_obs:
            seg['obs'] = (seg['obs']*255).to(torch.uint8)
        self.segs.append(seg)
        index_array = np.repeat(np.expand_dims(np.arange(seg['obs'].shape[0]-1), axis=0), seg['obs'].shape[1], axis=0)
        mask = index_array < step.cpu().numpy()[:, np.newaxis]
        self.indices_to_sample.append(mask)

    def feed_forward_generator(self, num_mini_batch=None, mini_batch_size=None):
        num_processes = self.segs[0]["obs"].size(1)
        num_steps = self.segs[0]["obs"].size(0) - 1
        obs_shape = self.segs[0]["obs"].shape[2:]
        num_segs = len(self.segs)

        if mini_batch_size is None:
            mini_batch_size = num_steps * (num_processes // num_mini_batch)

        indices_to_sample = np.array(self.indices_to_sample, dtype=np.uint8).transpose((0,2,1)).flatten()
        sampler = BatchSampler(
            WeightedRandomSampler(indices_to_sample, int(sum(indices_to_sample)), replacement=False), 
            mini_batch_size, 
            drop_last=True
        )

        all_obs = torch.concat([seg['obs'][:-1] for seg in self.segs], dim=0).view(-1, *obs_shape)
        all_returns = torch.concat([seg['returns'][:-1] for seg in self.segs], dim=0).view(-1, 1)

        for indices in sampler:
            obs_batch = all_obs[indices].to(self.device)
            returns_batch = all_returns[indices].to(self.device)

            # obs = []
            # returns = []

            # for idx in indices:
            #     process_idx = idx // num_segs
            #     seg_idx = idx % num_segs

            #     seg = self.segs[seg_idx]
            #     obs.append(seg["obs"][:, process_idx])
            #     returns.append(seg["returns"][:, process_idx])

            # obs = torch.stack(obs, dim=1).to(self.device)
            # returns = torch.stack(returns, dim=1).to(self.device)

            # obs_batch = obs[:-1].view(-1, *obs.size()[2:])
            # returns_batch = returns[:-1].view(-1, 1)

            if self.store_unnormalised_obs:
                obs_batch = obs_batch.to(torch.float32) / 255.

            yield obs_batch, returns_batch


class DCPG:
    """
    Delayed Critic Policy Gradient (DCPG)
    """

    def __init__(
        self,
        actor_critic,
        # PPO params
        clip_param,
        ppo_epoch,
        num_mini_batch,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        # Aux params
        buffer_size=None,
        aux_epoch=None,
        aux_freq=None,
        aux_num_mini_batch=None,
        policy_dist_coef=None,
        value_dist_coef=None,
        # Misc
        device=None,
        **kwargs
    ):
        # Actor-critic
        self.actor_critic = actor_critic

        # PPO params
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Aux params
        self.buffer = Buffer(buffer_size=buffer_size, device=device)

        self.aux_epoch = aux_epoch
        self.aux_freq = aux_freq
        self.aux_num_mini_batch = aux_num_mini_batch

        self.policy_dist_coef = policy_dist_coef
        self.value_dist_coef = value_dist_coef

        self.num_policy_updates = 0

        self.prev_value_loss_epoch = 0
        self.prev_policy_dist_epoch = 0

        # Optimizers
        self.policy_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.aux_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        # Add obs and returns to buffer
        seg = {key: rollouts[key].cpu() for key in ["obs", "returns"]}
        self.buffer.insert(seg, rollouts.step)

        # PPO phase
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        value_dist_epoch = 0

        # Clone actor-critic
        old_actor_critic = copy.deepcopy(self.actor_critic)

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                num_mini_batch=self.num_mini_batch
            )

            for sample in data_generator:
                # Sample batch
                (
                    obs_batch,
                    _,
                    actions_batch,
                    old_action_log_probs_batch,
                    _,
                    _,
                    _,
                    _,
                    adv_targs,
                    _,
                ) = sample

                # Feed batch to actor-critic
                actor_outputs, critic_outputs = self.actor_critic(obs_batch)
                dists = actor_outputs["dist"]
                values = critic_outputs["value"]

                # Feed batch to old actor-critic
                with torch.no_grad():
                    old_critic_outputs = old_actor_critic.forward_critic(obs_batch)
                old_values = old_critic_outputs["value"]

                # Compute action loss
                action_log_probs = dists.log_probs(actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                ratio_clipped = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surr1 = ratio * adv_targs
                surr2 = ratio_clipped * adv_targs
                action_loss = -torch.min(surr1, surr2).mean()
                dist_entropy = dists.entropy().mean()

                # Compute value dist
                value_dist = 0.5 * (values - old_values).pow(2).mean()

                # Update parameters  
                self.policy_optimizer.zero_grad()
                loss = (
                    action_loss
                    - dist_entropy * self.entropy_coef
                    + value_dist * self.value_dist_coef
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.policy_optimizer.step()

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                value_dist_epoch += value_dist.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        value_dist_epoch /= num_updates

        self.num_policy_updates += 1

        # Aux phase
        if self.num_policy_updates % self.aux_freq == 0:
            value_loss_epoch = 0
            policy_dist_epoch = 0

            # Clone actor-critic
            old_actor_critic = copy.deepcopy(self.actor_critic)

            for _ in range(self.aux_epoch):
                data_generator = self.buffer.feed_forward_generator(
                    num_mini_batch=self.aux_num_mini_batch
                )

                for sample in data_generator:
                    # Sample batch
                    obs_batch, returns_batch = sample

                    # Feed batch to actor-critic
                    actor_outputs, critic_outputs = self.actor_critic(obs_batch)
                    dists = actor_outputs["dist"]
                    values = critic_outputs["value"]

                    # Feed batch to old actor-critic
                    with torch.no_grad():
                        old_actor_outputs = old_actor_critic.forward_actor(obs_batch)
                    old_dists = old_actor_outputs["dist"]

                    # Compute value loss
                    value_loss = 0.5 * (values - returns_batch).pow(2).mean()

                    # Compute policy dist
                    policy_dist = kl_divergence(old_dists, dists).mean()

                    # Update parameters
                    self.aux_optimizer.zero_grad()
                    loss = value_loss + policy_dist * self.policy_dist_coef
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                    self.aux_optimizer.step()

                    value_loss_epoch += value_loss.item()
                    policy_dist_epoch += policy_dist.item()

            num_updates = self.aux_epoch * self.aux_num_mini_batch * len(self.buffer)
            value_loss_epoch /= num_updates
            policy_dist_epoch /= num_updates

            self.prev_value_loss_epoch = value_loss_epoch
            self.prev_policy_dist_epoch = policy_dist_epoch
        else:
            value_loss_epoch = self.prev_value_loss_epoch
            policy_dist_epoch = self.prev_policy_dist_epoch

        train_statistics = {
            "action_loss": action_loss_epoch,
            "dist_entropy": dist_entropy_epoch,
            "value_loss": value_loss_epoch,
            "policy_dist": policy_dist_epoch,
            "value_dist": value_dist_epoch,
        }

        return train_statistics
