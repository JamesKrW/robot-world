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
    def __init__(self, n_bins: int = 256, bin_path: str = None, 
                 strategy: Literal['uniform', 'quantile', 'kmeans', 'gaussian'] = 'uniform',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_bins = n_bins
        self.bins = None 
        self.min_vals = None
        self.max_vals = None
        self.bin_edges = None
        self.strategy = strategy
        self.bin_centers = None
        self.device = device

        if bin_path is not None and os.path.exists(bin_path):
            self.load_bins(bin_path)

    def save_bins(self, path: str) -> None:
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("No bins to save. Run create_bins first.")
            
        bin_data = {
            'n_bins': self.n_bins,
            'min_vals': self.min_vals.cpu().tolist(),
            'max_vals': self.max_vals.cpu().tolist(),
            'bin_edges': [edges.cpu().tolist() for edges in self.bin_edges],
            'strategy': self.strategy,
            'bin_centers': [centers.cpu().tolist() for centers in self.bin_centers],
            'version': '1.1'
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(bin_data, f)

    def load_bins(self, path: str) -> None:
        with open(path, 'r') as f:
            bin_data = json.load(f)
                
        self.n_bins = bin_data['n_bins']
        self.min_vals = torch.tensor(bin_data['min_vals'], device=self.device)
        self.max_vals = torch.tensor(bin_data['max_vals'], device=self.device)
        self.bin_edges = [torch.tensor(edges, device=self.device) for edges in bin_data['bin_edges']]
        self.strategy = bin_data['strategy']
        self.bin_centers = [torch.tensor(centers, device=self.device) for centers in bin_data['bin_centers']]

    def _create_bins_for_dimension(self, data: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.strategy == 'uniform':
            edges = torch.linspace(data.min(), data.max() + 1e-6, self.n_bins + 1, device=self.device)
            
        elif self.strategy == 'quantile':
            edges = torch.unique(torch.quantile(data, torch.linspace(0, 1, self.n_bins + 1, device=self.device)))
            if len(edges) < self.n_bins + 1:
                missing = self.n_bins + 1 - len(edges)
                extra_edges = torch.linspace(edges[-1], edges[-1] + 1e-6, missing + 1, device=self.device)[1:]
                edges = torch.cat([edges, extra_edges])
            
        elif self.strategy == 'kmeans':
            edges_np = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='kmeans').fit(data.cpu().reshape(-1, 1)).bin_edges_[0]
            edges = torch.tensor(edges_np, device=self.device)
            if len(edges) < self.n_bins + 1:
                missing = self.n_bins + 1 - len(edges)
                extra_edges = torch.linspace(edges[-1], edges[-1] + 1e-6, missing + 1, device=self.device)[1:]
                edges = torch.cat([edges, extra_edges])
            
        elif self.strategy == 'gaussian':
            mean, std = data.mean(), data.std()
            norm = torch.distributions.Normal(mean, std)
            edges = norm.icdf(torch.linspace(0.001, 0.999, self.n_bins + 1, device=self.device))
            edges[0], edges[-1] = data.min(), data.max() + 1e-6
            
        centers = (edges[1:] + edges[:-1]) / 2
        return edges, centers

    def collect_statistics(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        all_actions = []
        for batch in tqdm(dataloader):
            all_actions.append(batch["delta_action"])
        all_actions = torch.cat(all_actions, dim=0).to(self.device)
        
        min_vals = all_actions.min(dim=0)[0]
        max_vals = all_actions.max(dim=0)[0]
        
        bin_edges = []
        bin_centers = []
        
        for dim in range(7):
            edges, centers = self._create_bins_for_dimension(all_actions[:, dim], dim)
            bin_edges.append(edges)
            bin_centers.append(centers)
            
        self.bin_centers = bin_centers
        return min_vals, max_vals, bin_edges

    def convert_to_onehot(self, delta_actions: torch.Tensor) -> torch.Tensor:
        if self.bin_edges is None:
            raise ValueError("Bins not created yet")
            
        batch_size = delta_actions.shape[0]
        one_hot = torch.zeros((batch_size, 7, self.n_bins), device=delta_actions.device)
        
        for dim in range(7):
            indices = torch.bucketize(delta_actions[:, dim], self.bin_edges[dim]) - 1
            indices = torch.clamp(indices, 0, self.n_bins - 1)
            one_hot[torch.arange(batch_size, device=delta_actions.device), dim, indices] = 1
        one_hot = one_hot.view(batch_size, -1)
        return one_hot

    def decode_onehot(self, one_hot: torch.Tensor) -> torch.Tensor:
        if self.bin_edges is None:
            raise ValueError("Bins not created yet")
            
        batch_size = one_hot.shape[0] 
        delta_actions = torch.zeros((batch_size, 7), device=one_hot.device)
        
        for dim in range(7):
            indices = torch.argmax(one_hot[:, dim], dim=1)
            delta_actions[:, dim] = self.bin_centers[dim][indices]
            
        return delta_actions