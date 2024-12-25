import torch
import torch.nn as nn
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
import time
from tabulate import tabulate
import numpy as np
from contextlib import contextmanager
import gc
from typing import Dict, List, Tuple
import psutil
import os
import sys
from vae_new import VAE_models
def get_model_size(model: nn.Module) -> Tuple[int, float]:
    """Calculate model size in parameters and MB"""
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return param_count, size_mb

def get_dummy_input(batch_size: int, height: int, width: int, device: str) -> torch.Tensor:
    """Generate dummy input tensor"""
    return torch.randn(batch_size, 3, height, width, device=device)

class MemoryTracker:
    def __init__(self):
        self.max_mem = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.max_mem = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        else:
            self.max_mem = 0

def test_vae_config(
    model_name: str,
    model,
    batch_size: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """Test a VAE configuration and return metrics"""
    model = model.to(device)
    model.eval()
    
    # Get model statistics
    param_count, model_size_mb = get_model_size(model)
    
    # Generate dummy input
    dummy_input = get_dummy_input(
        batch_size,
        model.input_height,
        model.input_width,
        device
    )
    
    # Warm-up run
    with torch.no_grad():
        model(dummy_input)
    
    # Measure inference time and memory
    memory_tracker = MemoryTracker()
    with torch.no_grad(), memory_tracker:
        start_time = time.time()
        reconstructed, posterior = model(dummy_input)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
    max_mem = memory_tracker.max_mem
    
    # Get latent space statistics
    latent = posterior.sample()
    
    # Calculate reconstruction error
    rec_error = nn.MSELoss()(reconstructed, dummy_input).item()
    
    # Get latent space statistics
    latent_stats = {
        'mean': latent.mean().item(),
        'std': latent.std().item(),
        'min': latent.min().item(),
        'max': latent.max().item()
    }
    
    return {
        'model_name': model_name,
        'param_count': param_count,
        'model_size_mb': model_size_mb,
        'inference_time_ms': inference_time,
        'max_memory_mb': max_mem,
        'reconstruction_error': rec_error,
        'latent_stats': latent_stats,
        'input_shape': list(dummy_input.shape),
        'output_shape': list(reconstructed.shape),
        'latent_shape': list(latent.shape)
    }

def analyze_vae_configs(
    batch_sizes: List[int] = [1, 4, 8],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """Analyze different VAE configurations"""
    
    # Print system info
    print("\nSystem Information:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    results = []
    
    # Test each model configuration
    for model_name in VAE_models.keys():
        print(f"\nTesting {model_name}...")
        
        # Test with different batch sizes
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Create new model instance
            model = VAE_models[model_name]()
            
            try:
                # Run tests
                metrics = test_vae_config(
                    f"{model_name}_bs{batch_size}",
                    model,
                    batch_size,
                    device
                )
                results.append(metrics)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠️ Out of memory error with batch size {batch_size}")
                    # Clear memory
                    if device == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise e
            
            # Clean up
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Print results in tables
    print("\nModel Architecture Comparison:")
    arch_table = []
    for r in results:
        if r['input_shape'][0] == 1:  # Only show results for batch_size=1
            arch_table.append([
                r['model_name'],
                f"{r['param_count']:,}",
                f"{r['model_size_mb']:.1f}",
                f"{r['inference_time_ms']:.1f}",
                f"{r['max_memory_mb']:.1f}",
                f"{r['reconstruction_error']:.6f}"
            ])
    
    print(tabulate(
        arch_table,
        headers=['Model', 'Parameters', 'Size (MB)', 'Inference (ms)',
                'Max Memory (MB)', 'Rec. Error'],
        tablefmt='grid'
    ))
    
    # Print batch size scaling
    print("\nBatch Size Scaling (Inference Time ms):")
    batch_table = []
    for model_name in VAE_models.keys():
        row = [model_name]
        for bs in batch_sizes:
            matching_results = [r for r in results 
                              if r['model_name'] == f"{model_name}_bs{bs}"]
            if matching_results:
                row.append(f"{matching_results[0]['inference_time_ms']:.1f}")
            else:
                row.append("OOM")
        batch_table.append(row)
    
    print(tabulate(
        batch_table,
        headers=['Model'] + [f'BS={bs}' for bs in batch_sizes],
        tablefmt='grid'
    ))
    
    # Print latent space statistics
    print("\nLatent Space Statistics (Batch Size=1):")
    latent_table = []
    for r in results:
        if r['input_shape'][0] == 1:
            stats = r['latent_stats']
            latent_table.append([
                r['model_name'],
                f"{stats['mean']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                'x'.join(map(str, r['latent_shape']))
            ])
    
    print(tabulate(
        latent_table,
        headers=['Model', 'Mean', 'Std', 'Min', 'Max', 'Latent Shape'],
        tablefmt='grid'
    ))

if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Run analysis
    analyze_vae_configs(
        batch_sizes=[1, 4, 8],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )