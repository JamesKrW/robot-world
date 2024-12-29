from utils.action_binner import ActionBinner
from dataloader.dit_dataloader import create_diffusion_dataloader
import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_distributions(all_actions: np.ndarray, save_dir: str):
    """Plot histograms of action distributions"""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for dim in range(7):
        dim_data = all_actions[:, dim]
        
        # Calculate statistics
        mean = np.mean(dim_data)
        std = np.std(dim_data)
        skewness = stats.skew(dim_data)
        kurtosis = stats.kurtosis(dim_data)
        
        # Plot histogram with KDE
        sns.histplot(
            dim_data,
            bins=100,
            kde=True,
            ax=axes[dim]
        )
        
        axes[dim].set_title(f'Dimension {dim} Distribution')
        axes[dim].set_xlabel('Delta Action Value')
        axes[dim].set_ylabel('Count')
        
        # Add statistics annotation
        stats_text = (f'Mean: {mean:.4f}\n'
                     f'Std: {std:.4f}\n'
                     f'Skew: {skewness:.4f}\n'
                     f'Kurt: {kurtosis:.4f}')
        
        axes[dim].text(
            0.95, 0.95, stats_text,
            transform=axes[dim].transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    # Remove extra subplots
    axes[-1].remove()
    axes[-2].remove()
    
    plt.tight_layout()
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

def analyze_binning_strategies(dataloader, save_dir="binning_analysis"):
    """Analyze different binning strategies and save results"""
    os.makedirs(save_dir, exist_ok=True)
    strategies = ['uniform', 'quantile', 'kmeans', 'gaussian']
    results = {}
    
    # First collect all actions for distribution analysis
    all_actions = []
    print("Collecting all actions for initial analysis...")
    for _, _, delta_action in dataloader:
        all_actions.append(delta_action.numpy())
    all_actions = np.concatenate(all_actions, axis=0)
    
    # Plot distributions
    print("Plotting action distributions...")
    plot_distributions(all_actions, save_dir)
    
    # Analyze each strategy
    for strategy in strategies:
        try:
            print(f"\nTesting {strategy} binning strategy...")
            binner = ActionBinner(n_bins=256, strategy=strategy)
            
            # Collect statistics and create bins
            min_vals, max_vals, bin_edges = binner.collect_statistics(dataloader)
            binner.min_vals = min_vals
            binner.max_vals = max_vals
            binner.bin_edges = bin_edges
            
            # Save bins
            binner.save_bins(os.path.join(save_dir, f"bins_{strategy}.json"))
            
            # Test reconstruction error
            errors = []
            bin_usage = np.zeros((7, 256))  # Track bin usage
            errors_by_dim = [[] for _ in range(7)]  # Track errors by dimension
            
            for _, _, delta_actions in dataloader:
                # Convert to one-hot and back
                one_hot = binner.convert_to_onehot(delta_actions)
                reconstructed = binner.decode_onehot(one_hot)
                
                # Calculate error
                error = torch.abs(delta_actions - reconstructed)
                errors.append(error.cpu().numpy())
                
                # Track errors by dimension
                error_np = error.cpu().numpy()
                for dim in range(7):
                    errors_by_dim[dim].append(error_np[:, dim])
                
                # Track bin usage
                for dim in range(7):
                    bin_indices = np.clip(
                        np.digitize(delta_actions[:, dim].cpu().numpy(), binner.bin_edges[dim]) - 1,
                        0, 255
                    )
                    unique, counts = np.unique(bin_indices, return_counts=True)
                    bin_usage[dim, unique] += counts
            
            # Plot bin usage
            plot_bin_usage(bin_usage, strategy, save_dir)
            
            # Aggregate results
            errors = np.concatenate(errors)
            errors_by_dim = [np.concatenate(dim_errors) for dim_errors in errors_by_dim]
            
            results[strategy] = {
                'mean_error': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'median_error': float(np.median(errors)),
                'unused_bins': int(np.sum(bin_usage == 0)),
                'bin_usage_stats': {
                    'min_usage': int(np.min(bin_usage[bin_usage > 0])),
                    'max_usage': int(np.max(bin_usage)),
                    'mean_usage': float(np.mean(bin_usage[bin_usage > 0])),
                    'total_samples': int(np.sum(bin_usage))
                },
                'errors_by_dimension': {
                    f'dim_{i}': {
                        'mean': float(np.mean(dim_errors)),
                        'max': float(np.max(dim_errors)),
                        'median': float(np.median(dim_errors))
                    } for i, dim_errors in enumerate(errors_by_dim)
                },
                'bin_usage': bin_usage.tolist()
            }
            
            print(f"Results for {strategy}:")
            print(f"  Mean error: {results[strategy]['mean_error']:.6f}")
            print(f"  Max error: {results[strategy]['max_error']:.6f}")
            print(f"  Median error: {results[strategy]['median_error']:.6f}")
            print(f"  Unused bins: {results[strategy]['unused_bins']} / {7 * 256}")
            print("\n  Per-dimension mean errors:")
            for dim in range(7):
                print(f"    Dim {dim}: {results[strategy]['errors_by_dimension'][f'dim_{dim}']['mean']:.6f}")
            
        except Exception as e:
            print(f"Error analyzing {strategy} strategy: {str(e)}")
            results[strategy] = {'error': str(e)}
    
    # Save analysis results
    with open(os.path.join(save_dir, "analysis_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary comparison
    print("\nStrategy Comparison Summary:")
    print("-" * 60)
    print(f"{'Strategy':10} {'Mean Error':12} {'Median Error':12} {'Unused Bins':12}")
    print("-" * 60)
    for strategy in strategies:
        if 'error' not in results[strategy]:
            print(f"{strategy:10} {results[strategy]['mean_error']:12.6f} "
                  f"{results[strategy]['median_error']:12.6f} "
                  f"{results[strategy]['unused_bins']:12d}")
    
    return results

def main():
    # Create dataloader
    dataloader = create_diffusion_dataloader(
        batch_size=32,
        num_workers=4,
        data_dir="/home/kangrui/projects/world_model/droid-debug",
        split="train",
        dataset_name="droid_100"
    )
    
    # Analyze different binning strategies
    results = analyze_binning_strategies(dataloader)
    
    print("\nAnalysis complete. Check the binning_analysis directory for detailed results.")

if __name__ == "__main__":
    main()