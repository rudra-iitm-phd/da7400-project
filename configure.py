from state_embedding import StateEmbedding, EnergyEmbedding
from actor import Policy, ContinuousPolicy
import shared
import gymnasium as gym

class Configure:
      def __init__(self, script):
            
            self.script = script
            self.embedding = {
                  "vanilla":StateEmbedding,
                  "energy":EnergyEmbedding
            }

            self.policy = {

                  "d":Policy,
                  "c":ContinuousPolicy
            }


      def get_embedding(self):
            return self.embedding[self.script["embedding"]]

      def get_env(self):
            return gym.make(self.script["env"])

      def get_policy(self):
            if shared.ENV_NAME in ["BipedalWalker-v3"]:
                  return ContinuousPolicy
            return Policy
            

      

