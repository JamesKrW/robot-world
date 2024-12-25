import torch
from collections import defaultdict
import pandas as pd
from typing import Dict, Any
import numpy as np
from dit import DiT_models
def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_layers(model: torch.nn.Module) -> Dict[str, Any]:
    """Analyze different components of the DiT model."""
    stats = defaultdict(int)
    
    # Analyze each named parameter
    for name, param in model.named_parameters():
        # Get the module hierarchy
        parts = name.split('.')
        base_layer = parts[0]
        
        # Count parameters
        stats[f"{base_layer}_params"] += param.numel()
        
        # Track shapes
        if "weight" in name or "bias" in name:
            stats[f"{base_layer}_shapes"] = stats.get(f"{base_layer}_shapes", [])
            stats[f"{base_layer}_shapes"].append(f"{name}: {tuple(param.shape)}")
            
    return dict(stats)

def print_model_analysis(model_name: str = "DiT-S/2"):
    """Print detailed analysis of the DiT model."""
    # Create model
    model = DiT_models[model_name]()
    
    print(f"\n{'='*20} {model_name} Analysis {'='*20}")
    
    # 1. Overall model size
    total_params = count_parameters(model)
    print(f"\nTotal Parameters: {total_params:,}")
    
    # 2. Layer-wise analysis
    stats = analyze_model_layers(model)
    
    # Print parameter counts by component
    print("\nParameter Distribution:")
    param_counts = {k: v for k, v in stats.items() if k.endswith('_params')}
    total = sum(param_counts.values())
    
    for component, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
        base_name = component.replace('_params', '')
        percentage = (count / total) * 100
        print(f"{base_name:15s}: {count:,} parameters ({percentage:.1f}%)")
    
    # 3. Shape analysis
    print("\nLayer Shapes:")
    for component, shapes in stats.items():
        if component.endswith('_shapes'):
            base_name = component.replace('_shapes', '')
            print(f"\n{base_name}:")
            for shape in shapes:
                print(f"  {shape}")

    # 4. Memory estimation
    memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nApproximate Model Size in Memory: {memory_bytes / 1024 / 1024:.1f} MB")

import argparse
if __name__ == "__main__":
    # Analyze default model
    # add argument model_name to specify the model to analyze
    parser = argparse.ArgumentParser(description='Analyze DiT model')
    parser.add_argument('--model_name', type=str, default='DiT-S/2', help='DiT model name')
    args = parser.parse_args()
    model_name=args.model_name
    print_model_analysis(model_name)
    
    # Test model with different input sizes
    model = DiT_models[model_name]()
    
    print("\nTesting model with sample input:")
    batch_size = 2
    time_frames = 8
    channels = 16
    height = 18
    width = 32
    
    # Create sample inputs
    x = torch.randn(batch_size, time_frames, channels, height, width)
    t = torch.randint(0, 1000, (batch_size, time_frames))
    external_cond = torch.randn(batch_size, time_frames, 25)  # Example external condition
    
    # Run forward pass
    with torch.no_grad():
        output = model(x, t, external_cond)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Memory analysis during forward pass
    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        t = t.cuda()
        external_cond = external_cond.cuda()
        
        with torch.no_grad():
            output = model(x, t, external_cond)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\nPeak GPU memory usage: {peak_memory:.1f} MB")