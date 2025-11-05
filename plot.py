import torch 
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import wandb
import shared

def plot_state_representations_w_corr(rep_net, value_net, states,
                               device="cpu", method="pca",
                               save_dir="./plots", filename="plot.png",
                               show=False, log:bool=False):
      """
      Args:
            rep_net: torch.nn.Module mapping states -> representation φ(s)
            value_net: torch.nn.Module mapping representation -> V(φ(s))
            states: torch.Tensor of shape (batch_size, state_dim)
            device: str, "cpu" or "cuda"
            method: "pca" or "tsne" for dimensionality reduction
            save_dir: directory where the plot will be saved
            filename: name of the saved file (e.g., "plot1.png")
            show: if True, display the plot with plt.show()
      """
      rep_net.eval()
      value_net.eval()

      states = states

      with torch.no_grad():
            # Compute state representations
            reps = rep_net(states).cpu().numpy()

            # Dimensionality reduction for visualization
            if reps.shape[1] > 2:
                  if method == "pca":
                        reducer = PCA(n_components=3)  # keep 3 PCs for correlation
                  elif method == "tsne":
                        reducer = TSNE(n_components=2, perplexity=30, learning_rate="auto")
                  else:
                        raise ValueError("method must be 'pca' or 'tsne'")
                  reps_reduced = reducer.fit_transform(reps)
            else:
                  reps_reduced = reps

            # Compute values
            value_net.to("cpu")
            values = value_net(torch.tensor(reps)).squeeze().cpu().numpy()
            value_net.to(device)

      if method == "pca":
            # --- Create figure with 1 main scatter + 3 regression plots ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # --- Top-left: scatter of state reps colored by value ---
            scatter = axes[0, 0].scatter(
                  reps_reduced[:, 0], reps_reduced[:, 1],
                  c=values,
                  cmap="viridis",
                  s=40,
                  alpha=0.8
            )
            fig.colorbar(scatter, ax=axes[0, 0], label="$V(\phi(s))$")
            axes[0, 0].set_xlabel("PC1")
            axes[0, 0].set_ylabel("PC2")
            axes[0, 0].set_title(f"State Representations (PCA) Colored by Value, Env :{shared.ENV_NAME}")

            # --- Regression plots for PC1, PC2, PC3 ---
            for i, ax in enumerate([axes[0, 1], axes[1, 0], axes[1, 1]]):
                  pc = reps_reduced[:, i]
                  slope, intercept = np.polyfit(pc, values, 1)
                  reg_line = slope * pc + intercept

                  ax.scatter(pc, values, alpha=0.6, label="Data")
                  ax.plot(pc, reg_line, color="red", label=f"y={slope:.2f}x+{intercept:.2f}")
                  ax.set_xlabel(f"PC{i+1}")
                  ax.set_ylabel("Value")
                  ax.set_title(f"PC{i+1} vs Value with Regression")
                  ax.legend()
            # if log:
            #       wandb.log({"State Embeddings": [wandb.Image(fig)]})

      else:
            # Just show the t-SNE scatter (no PC correlations)
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(
                  reps_reduced[:, 0], reps_reduced[:, 1],
                  c=values,
                  cmap="viridis",
                  s=40,
                  alpha=0.8
            )
            fig.colorbar(scatter, ax=ax, label="$V(\phi(s))$")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_title("State Representations (t-SNE) Colored by Value")
      plt.close()
      return fig

      