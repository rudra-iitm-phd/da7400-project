import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

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
      """Simple continuous policy with reparametrization trick"""
      def __init__(self, feature_dim, action_dim, hidden_dim=256):
            super(ContinuousPolicy, self).__init__()

            self.feature_dim = feature_dim
            self.action_dim = action_dim
            
            # Simple 2-layer network
            self.net = nn.Sequential(
                  nn.Linear(feature_dim, hidden_dim),
                  nn.ReLU(),
                  nn.Linear(hidden_dim, hidden_dim),
                  nn.ReLU()
            )
            
            # Output layers
            self.mean_layer = nn.Linear(hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(hidden_dim, action_dim)
            
            # Log std bounds
            self.log_std_min = -20
            self.log_std_max = 2

            self.apply(self._weights_init)

      def forward(self, state):
            """Returns mean and std for the action distribution"""
            x = self.net(state)
            mean = self.mean_layer(x)
            log_std = self.log_std_layer(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            return mean, std

      def sample_action(self, state):
            """
            Sample action using reparametrization trick.
            Returns: (action, log_prob) both with gradients for backprop
            """
            mean, std = self.forward(state)
            
            # Create normal distribution
            normal = Normal(mean, std)
            
            # Sample using reparametrization trick (rsample allows gradients)
            z = normal.rsample()
            
            # Apply tanh to bound actions to [-1, 1]
            action = torch.tanh(z)
            
            # Compute log probability
            log_prob = normal.log_prob(z)
            
            # Apply tanh correction to log_prob
            # log π(a|s) = log μ(z|s) - log(1 - tanh²(z))
            log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return action, log_prob

      def act(self, state, deterministic=False):
            """Get action for environment interaction (no gradients)"""
            with torch.no_grad():
                  mean, std = self.forward(state)
                  
                  if deterministic:
                        action = torch.tanh(mean)
                  else:
                        normal = Normal(mean, std)
                        z = normal.sample()
                        action = torch.tanh(z)
                  
                  return action.cpu().numpy()

      def _weights_init(self, m):
            if isinstance(m, nn.Linear):
                  nn.init.orthogonal_(m.weight, gain=1.0)
                  nn.init.constant_(m.bias, 0)
      