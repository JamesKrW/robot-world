from tqdm import tqdm
from typing import Tuple, List, Literal
import os
import json
import warnings
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning

class ActionBinner:
    def __init__(
        self, 
        n_bins: int = 256, 
        bin_path: str = None,
        strategy: Literal['uniform', 'quantile', 'kmeans', 'gaussian'] = 'uniform'
    ):
        self.n_bins = n_bins
        self.bins = None
        self.min_vals = None
        self.max_vals = None
        self.bin_edges = None
        self.strategy = strategy
        self.bin_centers = None
        
        # Try to load existing bins if path is provided
        if bin_path is not None and os.path.exists(bin_path):
            self.load_bins(bin_path)
    
    def save_bins(self, path: str) -> None:
        """Save bin information to file"""
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("No bins to save. Run create_bins first.")
            
        bin_data = {
            'n_bins': self.n_bins,
            'min_vals': self.min_vals.tolist(),
            'max_vals': self.max_vals.tolist(),
            'bin_edges': [edges.tolist() for edges in self.bin_edges],
            'strategy': self.strategy,
            'bin_centers': [centers.tolist() for centers in self.bin_centers],
            'version': '1.1'  # Added version tracking
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(bin_data, f)
        print(f"Saved bin data to {path}")
    
    def load_bins(self, path: str) -> None:
        """Load bin information from file"""
        print(f"Loading bin data from {path}")
        try:
            with open(path, 'r') as f:
                bin_data = json.load(f)
                
            self.n_bins = bin_data['n_bins']
            self.min_vals = np.array(bin_data['min_vals'])
            self.max_vals = np.array(bin_data['max_vals'])
            self.bin_edges = [np.array(edges) for edges in bin_data['bin_edges']]
            self.strategy = bin_data['strategy']
            self.bin_centers = [np.array(centers) for centers in bin_data['bin_centers']]
            print(f"Successfully loaded bin data using {self.strategy} strategy")
        except Exception as e:
            print(f"Error loading bins from {path}: {str(e)}")
            raise
    
    def _create_bins_for_dimension(self, data: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create bins for a single dimension using specified strategy"""
        if self.strategy == 'uniform':
            edges = np.linspace(np.min(data), np.max(data) + 1e-6, self.n_bins + 1)
            
        elif self.strategy == 'quantile':
            edges = np.unique(np.percentile(data, np.linspace(0, 100, self.n_bins + 1)))
            # If we got fewer unique edges than needed, add small offsets
            if len(edges) < self.n_bins + 1:
                missing = self.n_bins + 1 - len(edges)
                extra_edges = np.linspace(edges[-1], edges[-1] + 1e-6, missing + 1)[1:]
                edges = np.concatenate([edges, extra_edges])
            
        elif self.strategy == 'kmeans':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='kmeans')
                kbd.fit(data.reshape(-1, 1))
                edges = kbd.bin_edges_[0]
                
                # Handle case where KMeans finds fewer clusters
                if len(edges) < self.n_bins + 1:
                    missing = self.n_bins + 1 - len(edges)
                    extra_edges = np.linspace(edges[-1], edges[-1] + 1e-6, missing + 1)[1:]
                    edges = np.concatenate([edges, extra_edges])
            
        elif self.strategy == 'gaussian':
            mean, std = np.mean(data), np.std(data)
            edges = stats.norm.ppf(np.linspace(0.001, 0.999, self.n_bins + 1), mean, std)
            edges[0] = np.min(data)
            edges[-1] = np.max(data) + 1e-6
            
        else:
            raise ValueError(f"Unknown binning strategy: {self.strategy}")
        
        centers = (edges[1:] + edges[:-1]) / 2
        return edges, centers
    
    def collect_statistics(self, dataloader) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Collect statistics for binning from the dataloader"""
        all_actions = []
        print("Collecting delta action statistics...")
        
        for _, _, delta_action in tqdm(dataloader):
            all_actions.append(delta_action.numpy())
            
        all_actions = np.concatenate(all_actions, axis=0)  # Shape: (N, 7)
        
        # Basic statistics
        min_vals = np.min(all_actions, axis=0)
        max_vals = np.max(all_actions, axis=0)
        
        # Create bins based on selected strategy
        bin_edges = []
        bin_centers = []
        
        for dim in range(7):
            edges, centers = self._create_bins_for_dimension(all_actions[:, dim], dim)
            bin_edges.append(edges)
            bin_centers.append(centers)
            
        self.bin_centers = bin_centers
        return min_vals, max_vals, bin_edges
    
    def create_bins(self, min_vals: np.ndarray = None, max_vals: np.ndarray = None) -> List[np.ndarray]:
        """Create bins for each dimension"""
        if min_vals is None or max_vals is None:
            raise ValueError("min_vals and max_vals must be provided")
            
        self.min_vals = min_vals
        self.max_vals = max_vals
        
        if self.strategy != 'uniform':
            raise ValueError("For non-uniform binning strategies, use collect_statistics instead")
            
        bin_edges = []
        bin_centers = []
        for dim in range(7):
            edges = np.linspace(min_vals[dim], max_vals[dim] + 1e-6, self.n_bins + 1)
            centers = (edges[1:] + edges[:-1]) / 2
            bin_edges.append(edges)
            bin_centers.append(centers)
        
        self.bin_edges = bin_edges
        self.bin_centers = bin_centers
        return bin_edges
    
    def convert_to_onehot(self, delta_actions: torch.Tensor) -> torch.Tensor:
        """Convert batch of delta actions to one-hot encoded vectors"""
        if self.bin_edges is None:
            raise ValueError("Bins not created yet. Run create_bins first.")
            
        batch_size = delta_actions.shape[0]
        device = delta_actions.device
        
        # Move to CPU for numpy operations
        delta_actions_np = delta_actions.cpu().numpy()
        
        # Initialize output array
        one_hot = np.zeros((batch_size, 7, self.n_bins))
        
        # For each dimension
        for dim in range(7):
            # Find bin indices for each value in the batch
            indices = np.digitize(delta_actions_np[:, dim], self.bin_edges[dim]) - 1
            
            # Clip to ensure valid indices
            indices = np.clip(indices, 0, self.n_bins - 1)
            
            # Set one-hot values
            one_hot[np.arange(batch_size), dim, indices] = 1
            
        return torch.from_numpy(one_hot).float().to(device)
    
    def decode_onehot(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Convert one-hot encoded vectors back to approximate delta actions"""
        if self.bin_edges is None:
            raise ValueError("Bins not created yet. Run create_bins first.")
            
        device = one_hot.device
        batch_size = one_hot.shape[0]
        
        # Initialize output array
        delta_actions = torch.zeros((batch_size, 7), device=device)
        
        # For each dimension
        for dim in range(7):
            # Get indices of 1s in one-hot vectors
            indices = torch.argmax(one_hot[:, dim], dim=1)
            
            # Get corresponding bin center values
            centers = torch.tensor(self.bin_centers[dim], device=device)
            delta_actions[:, dim] = centers[indices]
            
        return delta_actions