"""Feed-forward actor-critic with a depth CNN over the leading part of the observation.

Includes small MLP on proprio before concat. Encoded features go directly into the actor and critic MLPs.

Observation layout:

- The vector must be [flat_image | proprio] where flat_image has length
  cnn_in_channels * img_h * img_w and is reshaped to (B, cnn_in_channels, img_h, img_w).
- Proprio width is num_actor_obs - cnn_in_channels * img_h * img_w (must match num_critic_obs).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic

from .depth_cnn import DepthCNN


class ActorCriticCNN(ActorCritic):

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        img_h: int = 32,
        img_w: int = 32,
        cnn_in_channels: int = 1,
        cnn_base_channels: int = 32,
        state_mlp_hidden_dim: int = 64,
        **kwargs,
    ):
        kwargs.pop("rnn_type", None)
        kwargs.pop("rnn_hidden_dim", None)
        kwargs.pop("rnn_hidden_size", None)
        kwargs.pop("rnn_num_layers", None)

        ih, iw = int(img_h), int(img_w)
        c_ic = int(cnn_in_channels)
        if c_ic < 1:
            raise ValueError(f"cnn_in_channels must be >= 1; got {c_ic}.")
        depth_flat_dim = ih * iw * c_ic
        na, nc = int(num_actor_obs), int(num_critic_obs)
        if na != nc:
            raise ValueError(
                f"ActorCriticCNN expects num_actor_obs == num_critic_obs; got {na} vs {nc}."
            )
        if na < depth_flat_dim:
            raise ValueError(
                f"num_actor_obs ({na}) must be >= cnn_in_channels*img_h*img_w ({depth_flat_dim}); "
                f"cnn_in_channels={c_ic}, img_h={ih}, img_w={iw}."
            )
        state_dim = na - depth_flat_dim
        if state_dim < 1:
            raise ValueError(
                f"Implied proprio width is {state_dim}; need at least one proprio dimension after the image block."
            )

        cnn = DepthCNN(in_channels=c_ic, base_channels=cnn_base_channels)

        self.state_dim = state_dim
        self.img_h = ih
        self.img_w = iw
        self.cnn_in_channels = c_ic
        self.depth_flat_dim = depth_flat_dim
        encoder_out_dim = cnn.out_dim + int(state_mlp_hidden_dim)

        state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, int(state_mlp_hidden_dim)),
            nn.ELU(),
        )

        super().__init__(
            num_actor_obs=encoder_out_dim,
            num_critic_obs=encoder_out_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        self.cnn = cnn
        self.state_mlp = state_mlp

        print(f"ActorCriticCNN encoder_out_dim={encoder_out_dim} (CNN + state MLP)")

    def split_obs(self, obs: torch.Tensor):
        B = obs.shape[0]

        depth = obs[:, : self.depth_flat_dim]
        state = obs[:, self.depth_flat_dim :]

        depth = depth.view(B, self.cnn_in_channels, self.img_h, self.img_w)

        return depth, state

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() not in (2, 3):
            raise ValueError(f"Unexpected obs shape {obs.shape}")
        leading, d = obs.shape[:-1], obs.shape[-1]
        flat = obs.reshape(-1, d)
        depth, state = self.split_obs(flat)
        img_feat = self.cnn(depth)
        state_feat = self.state_mlp(state)
        enc = torch.cat([img_feat, state_feat], dim=-1)
        return enc.reshape(*leading, -1)

    def act(self, observations, **kwargs):
        encoded = self.encode(observations)
        if encoded.dim() == 3:
            T, N, E = encoded.shape
            flat = encoded.reshape(T * N, E)
            self.update_distribution(flat)
            return self.distribution.sample().view(T, N, -1)
        self.update_distribution(encoded)
        return self.distribution.sample()

    def act_inference(self, observations):
        encoded = self.encode(observations)
        if encoded.dim() == 3:
            T, N, E = encoded.shape
            return self.actor(encoded.reshape(T * N, E)).view(T, N, -1)
        return self.actor(encoded)

    def evaluate(self, critic_observations, **kwargs):
        encoded = self.encode(critic_observations)
        if encoded.dim() == 3:
            T, N, E = encoded.shape
            return self.critic(encoded.reshape(T * N, E)).view(T, N, -1)
        return self.critic(encoded)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)
