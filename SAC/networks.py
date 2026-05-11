import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions,
                 fc1_dims=256, fc2_dims=256,
                 name='critic', chkpt_dir='tmp/sac'):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_sac.pt')

        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, n_actions=2,
                 fc1_dims=256, fc2_dims=256,
                 name='actor', chkpt_dir='tmp/sac'):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_sac.pt')

        self.n_actions = n_actions
        self.reparam_noise = 1e-2

        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)

        try:
            self.log_std.bias.data.fill_(-1.0)
        except Exception:
            pass

        max_action_t = T.tensor(max_action, dtype=T.float32)
        if max_action_t.ndim == 0:
            max_action_t = max_action_t.repeat(n_actions)
        self.register_buffer("max_action", max_action_t)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)

        mu = T.tanh(mu) * self.max_action
        log_std = T.clamp(log_std, min=-2.0, max=1.0)
        std = log_std.exp()
        std = std.clamp(min=1e-2, max=2.0)
        return mu, std, log_std

    def sample(self, state, reparameterize=True):
        mu, std, _ = self.forward(state)
        dist = Normal(mu, std)

        z = dist.rsample() if reparameterize else dist.sample()
        tanh_z = T.tanh(z)
        tanh_z = T.clamp(tanh_z, -0.95, 0.95)
        action = tanh_z * self.max_action

        log_prob_raw = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob_correction = T.sum(
            T.log(1 - tanh_z.pow(2) + self.reparam_noise),
            dim=-1,
            keepdim=True
        )
        log_prob = log_prob_raw - log_prob_correction

        return action, log_prob

    def deterministic(self, state):
        mu, _, _ = self.forward(state)
        return mu

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))