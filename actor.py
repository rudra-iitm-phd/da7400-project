import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Policy(nn.Module):
      def __init__(self, input_dim, n_actions):
            super(Policy, self).__init__()

            self.input_dim = input_dim
            self.output_dim = n_actions 

            self.policy_net = nn.Sequential(
                  nn.Linear(self.input_dim, self.output_dim)
            )

            self.apply(self._weights_init)

      def forward(self, state_embedding):
            action_logits = self.policy_net(state_embedding)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs

      def _weights_init(self, m):
            
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)

class ContinuousPolicy(nn.Module):
      def __init__(self, feature_dim, action_dim):
            super(ContinuousPolicy, self).__init__()

            self.feature_dim = feature_dim
            self.action_dim = action_dim
            self.log_std_min, self.log_std_max = -20, 2

            self.fc = nn.Sequential(
                  nn.Linear(self.feature_dim, self.feature_dim),
                  # nn.ReLU(),
                  # nn.Linear(self.feature_dim, self.feature_dim),
                  # nn.ReLU()
            )

            self.mean = nn.Linear(self.feature_dim, self.action_dim)
            self.log_std = nn.Linear(self.feature_dim, self.action_dim)

            self.apply(self._weights_init)

      def forward(self, state):
            x = self.fc(state)
            mean = self.mean(x)
            log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            return mean, std

      def sample(self, obs: torch.Tensor):
            """
            Sample an action using reparameterization trick.
            Returns action (tanh squashed) and log_prob (for training).
            """
            mean, std = self.forward(obs)
            dist = Normal(mean, std)
            
            # Reparameterization trick
            z = dist.rsample()
            
            # Tanh squashing
            action = torch.tanh(z)
            
            # Compute log probability with tanh transformation
            log_prob = dist.log_prob(z)
            
            # Apply correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
            
            return action, log_prob.squeeze()

      def log_prob(self, obs, actions):
            """
            Compute log probability of given actions.
            """
            mean, std = self.forward(obs)
            dist = Normal(mean, std)
            
            # Inverse tanh to get z
            z = torch.atanh(torch.clamp(actions, -1 + 1e-6, 1 - 1e-6))
            
            log_prob = dist.log_prob(z)
            log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
            return log_prob.sum(-1)

      def act(self, obs, deterministic=False):
            """
            Returns an action (numpy) suitable for env.step().
            If deterministic=True, returns tanh(mean).
            """
            if not torch.is_tensor(obs):
                  obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            mean, std = self.forward(obs)
            if deterministic:
                  action = torch.tanh(mean)
            else:
                  dist = Normal(mean, std)
                  z = dist.sample()
                  action = torch.tanh(z)

            return action.detach().cpu().numpy().squeeze()

      def _weights_init(self, m):
            
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)

      