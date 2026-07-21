"""Recurrent actor-critic with a depth CNN over the leading part of the observation.

Observation layout (required):

The vector must be [flat_image | proprio] where flat_image has length
  cnn_in_channels * img_h * img_w and is reshaped to
  (B, cnn_in_channels, img_h, img_w) for the CNN (depth only: cnn_in_channels=1;
  depth + obstacle mask: cnn_in_channels=2).
  img_h and img_w must match the env's downsampled grid (e.g.
  ElevationPolicyObsSpec.depth_downsample_hw).
  
Proprio width is
  num_actor_obs - cnn_in_channels * img_h * img_w (and must match num_critic_obs).
"""

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory

from .depth_cnn import DepthCNN


class ActorCriticRecurrentCNN(ActorCritic):
    """See module docstring: image stack first, spatial size (img_h, img_w) and cnn_in_channels explicit."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="relu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        img_h: int = 32,
        img_w: int = 32,
        cnn_in_channels: int = 1,
        cnn_base_channels: int = 32,
        state_mlp_hidden_dim: int = 64,
        **kwargs,
    ):
        ih, iw = int(img_h), int(img_w)
        c_ic = int(cnn_in_channels)
        if c_ic < 1:
            raise ValueError(f"cnn_in_channels must be >= 1; got {c_ic}.")
        depth_flat_dim = ih * iw * c_ic
        na, nc = int(num_actor_obs), int(num_critic_obs)
        if na != nc:
            raise ValueError(
                f"ActorCriticRecurrentCNN expects num_actor_obs == num_critic_obs; got {na} vs {nc}."
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
        
        self.encoder_out_dim = cnn.out_dim + int(state_mlp_hidden_dim)
        #self.encoder_out_dim = int(cnn_mlp_hidden_dim) + int(state_mlp_hidden_dim)

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        self.feature_norm = nn.LayerNorm(self.encoder_out_dim)
        self.cnn = cnn

        state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, int(state_mlp_hidden_dim)),
            nn.ELU(),
        )

        #cnn_mlp = nn.Sequential(
        #    nn.Linear(self.cnn.out_dim, int(cnn_mlp_hidden_dim) * 2),
        #    nn.ELU(),
        #    nn.Linear(int(cnn_mlp_hidden_dim) * 2, int(cnn_mlp_hidden_dim)),
        #    nn.ELU()
        #)

        #self.cnn_mlp = cnn_mlp
        self.state_mlp = state_mlp

        self.memory_a = Memory(
            self.encoder_out_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        self.memory_c = Memory(
            self.encoder_out_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def split_obs(self, obs):
        B = obs.shape[0]

        depth = obs[:, : self.depth_flat_dim]
        state = obs[:, self.depth_flat_dim :]

        depth = depth.view(B, self.cnn_in_channels, self.img_h, self.img_w)

        return depth, state

    def encode(self, obs):
        if obs.dim() == 2:
            depth, state = self.split_obs(obs)

            img_feat = self.cnn(depth)
            #img_feat = self.cnn_mlp(depth)
            state_feat = self.state_mlp(state)
            encoded = torch.cat([img_feat, state_feat], dim=-1)
            encoded = self.feature_norm(encoded)

        elif obs.dim() == 3:
            T, N, D = obs.shape
            x = obs.reshape(T * N, D)
            depth, state = self.split_obs(x)

            img_feat = self.cnn(depth)
            #img_feat = self.cnn_mlp(depth)
            state_feat = self.state_mlp(state)
            encoded = torch.cat([img_feat, state_feat], dim=-1)
            encoded = self.feature_norm(encoded)
            encoded = encoded.reshape(T, N, -1)
        else:
            raise ValueError(f"Unexpected obs shape {obs.shape}")

        return encoded

    def act(self, observations, masks=None, hidden_states=None):
        encoded = self.encode(observations)
        input_a = self.memory_a(encoded, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        encoded = self.encode(observations)
        input_a = self.memory_a(encoded)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        encoded = self.encode(critic_observations)
        input_c = self.memory_c(encoded, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states