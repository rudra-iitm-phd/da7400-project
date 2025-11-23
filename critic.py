import numpy as np 
import torch 
import torch.nn as nn 
import shared


class ValueNetwork(nn.Module):
      def __init__(self, embedding_dim):
            super(ValueNetwork, self).__init__()

            self.input_dim = embedding_dim

            if shared.ENV_NAME == "BipedalWalker-v3":
                  self.value = nn.Sequential(
                        nn.Linear(embedding_dim, 256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 128),
                        nn.LeakyReLU(),
                        nn.Linear(128, 1)
                  )

            else:
                  self.value = nn.Sequential(
                        nn.Linear(self.input_dim, 1)
                        
                  )

            self.apply(self._weights_init)

      def forward(self, state_embedding):

            return self.value(state_embedding)

      def _weights_init(self, m):
            
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)
                  # nn.init.constant_(m.bias, 0)