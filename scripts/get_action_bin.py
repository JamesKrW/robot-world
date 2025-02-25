from robot_world.utils.action_binner import ActionBinner
from robot_world.dataloader.dit_dataloader import create_diffusion_dataloader
import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_binning_strategies(dataloader, save_dir="binning_analysis"):
   os.makedirs(save_dir, exist_ok=True)
   strategies = ['kmeans', 'gaussian','uniform', 'quantile']
   results = {}
   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Collect actions
   all_actions = []
   for batch in dataloader:
       all_actions.append(batch['delta_action'])
   all_actions = torch.cat(all_actions, dim=0).to(device)

   # Plot distributions
   plot_distributions(all_actions.cpu(), save_dir)
   
   for strategy in strategies:
       print(f"\nTesting {strategy} binning strategy...")
       binner = ActionBinner(n_bins=256, strategy=strategy, device=device)
       
       min_vals, max_vals, bin_edges = binner.collect_statistics(dataloader)
       binner.min_vals = min_vals
       binner.max_vals = max_vals
       binner.bin_edges = bin_edges
       
       binner.save_bins(os.path.join(save_dir, f"bins_{strategy}.json"))
       
       # Test reconstruction
       errors = []
       bin_usage = torch.zeros((7, 256), device=device)
       errors_by_dim = [[] for _ in range(7)]
       
       for batch in dataloader:
           delta_actions = batch['delta_action'].to(device)
           one_hot = binner.convert_to_onehot(delta_actions)
           reconstructed = binner.decode_onehot(one_hot)
           
           error = torch.abs(delta_actions - reconstructed)
           errors.append(error)
           
           # Track errors by dimension
           for dim in range(7):
               errors_by_dim[dim].append(error[:, dim])
           
           # Track bin usage  
           for dim in range(7):
               indices = torch.bucketize(delta_actions[:, dim], binner.bin_edges[dim]) - 1
               indices = torch.clamp(indices, 0, 255)
               unique, counts = torch.unique(indices, return_counts=True) 
               bin_usage[dim, unique] += counts

       plot_bin_usage(bin_usage.cpu().numpy(), strategy, save_dir)
       
       # Aggregate results
       errors = torch.cat(errors)
       errors_by_dim = [torch.cat(dim_errors) for dim_errors in errors_by_dim]
       
       results[strategy] = {
           'mean_error': float(errors.mean().cpu()),
           'max_error': float(errors.max().cpu()),
           'median_error': float(errors.median().cpu()),
           'unused_bins': int((bin_usage == 0).sum().cpu()),
           'bin_usage_stats': {
               'min_usage': int(bin_usage[bin_usage > 0].min().cpu()),
               'max_usage': int(bin_usage.max().cpu()),
               'mean_usage': float(bin_usage[bin_usage > 0].mean().cpu()),
               'total_samples': int(bin_usage.sum().cpu())
           },
           'errors_by_dimension': {
               f'dim_{i}': {
                   'mean': float(dim_errors.mean().cpu()),
                   'max': float(dim_errors.max().cpu()),
                   'median': float(dim_errors.median().cpu())
               } for i, dim_errors in enumerate(errors_by_dim)
           },
           'bin_usage': bin_usage.cpu().tolist()
       }

       print(f"Results for {strategy}:")
       print(f"  Mean error: {results[strategy]['mean_error']:.6f}")
       print(f"  Max error: {results[strategy]['max_error']:.6f}")
       print(f"  Unused bins: {results[strategy]['unused_bins']}")

   with open(os.path.join(save_dir, "analysis_results.json"), 'w') as f:
       json.dump(results, f, indent=2)

   return results

def plot_distributions(all_actions: torch.Tensor, save_dir: str):
   os.makedirs(save_dir, exist_ok=True)
   fig, axes = plt.subplots(3, 3, figsize=(15, 15))
   axes = axes.flatten()
   
   for dim in range(7):
       dim_data = all_actions[:, dim]
       mean = dim_data.mean()
       std = dim_data.std()
       
       sns.histplot(
           dim_data.cpu(),
           bins=100,
           kde=True,
           ax=axes[dim]
       )
       
       axes[dim].set_title(f'Dimension {dim}')
       axes[dim].set_xlabel('Delta Action Value')
       axes[dim].text(
           0.95, 0.95, 
           f'Mean: {mean:.4f}\nStd: {std:.4f}',
           transform=axes[dim].transAxes,
           va='top', ha='right',
           bbox=dict(facecolor='white', alpha=0.8)
       )
   
   for ax in axes[-2:]:
       ax.remove()
   
   plt.savefig(os.path.join(save_dir, 'action_distributions.png'), dpi=300, bbox_inches='tight')
   plt.close()

   

def plot_bin_usage(bin_usage: np.ndarray, strategy: str, save_dir: str):
    """Plot bin usage heatmap"""
    plt.figure(figsize=(15, 5))
    
    # Log scale for better visualization
    plot_data = np.log1p(bin_usage)
    
    sns.heatmap(
        plot_data,
        cmap='viridis',
        xticklabels=50,  # Show every 50th bin
        yticklabels=['Dim ' + str(i) for i in range(7)],
        cbar_kws={'label': 'Log(Usage + 1)'}
    )
    
    plt.title(f'Bin Usage Distribution ({strategy})')
    plt.xlabel('Bin Index')
    plt.ylabel('Dimension')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'bin_usage_{strategy}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
   dataloader = create_diffusion_dataloader(
       batch_size=32,
       num_workers=4,
       data_dir="../datasets/droid_100",
       split="train",
       dataset_name="droid_100"
   )
   
   analyze_binning_strategies(dataloader)
   
if __name__ == "__main__":
    main()
