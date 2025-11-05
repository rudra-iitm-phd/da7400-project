from state_embedding import StateEmbedding, EnergyEmbedding
import gymnasium as gym

class Configure:
      def __init__(self, script):
            
            self.script = script
            self.embedding = {
                  "vanilla":StateEmbedding,
                  "energy":EnergyEmbedding
            }


      def get_embedding(self):
            return self.embedding[self.script["embedding"]]

      def get_env(self):
            return gym.make(self.script["env"])

