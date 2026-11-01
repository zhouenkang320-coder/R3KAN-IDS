import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from kan_layers import KAN


class KANActorCritic(nn.Module):

    def __init__(
        self,
        state_dim,
        hidden_dim=64,
        num_layers=2,
        grid_size=12,
        spline_order=3,
        critic_hidden_dim=128,
        input_grid_range=(-1.0, 1.0),
        hidden_grid_range=(-3.0, 3.0),
        use_hidden_norm=True,
    ):
        super().__init__()
        self.actor_backbone = KAN(
            in_features=state_dim,
            hidden_dim=hidden_dim,
            out_features=hidden_dim,
            num_layers=num_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            input_grid_range=input_grid_range,
            hidden_grid_range=hidden_grid_range,
            use_hidden_norm=use_hidden_norm,
        )
        self.actor_head = nn.Linear(hidden_dim, 2)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        self.critic_backbone = nn.Sequential(
            nn.Linear(state_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(critic_hidden_dim, 1)
        for module in self.critic_backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def forward(self, x):
        features, _, _ = self.actor_backbone(x)
        logits = torch.clamp(self.actor_head(features), -100, 100)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        value = self.critic_head(self.critic_backbone(x)).squeeze(-1)
        return (logits, torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return (action, distribution.log_prob(action), distribution.entropy(), value)

    def get_value(self, x):
        return self(x)[1]


def create_agent(config, feature_dim):
    return KANActorCritic(
        state_dim=feature_dim,
        hidden_dim=config.kan_hidden_dim,
        num_layers=config.kan_num_layers,
        grid_size=config.kan_grid_size,
        spline_order=config.kan_spline_order,
        critic_hidden_dim=config.critic_hidden_dim,
        input_grid_range=(-config.kan_input_grid_range, config.kan_input_grid_range),
        hidden_grid_range=(-config.kan_hidden_grid_range, config.kan_hidden_grid_range),
        use_hidden_norm=config.kan_use_hidden_norm,
    )
