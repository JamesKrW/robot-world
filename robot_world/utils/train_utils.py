import json
from pathlib import Path
from dataclasses import fields, MISSING
import argparse
from typing import Type, TypeVar, Any
import torch
import numpy as np
from dataclasses import dataclass
T = TypeVar('T')

@dataclass
class ConfigMixin:
    """Mixin class for configuration dataclasses providing common functionality"""
    
    def save(self, path: str):
        """Save config to json"""
        config_dict = self.__dict__.copy()
        
        # Convert any tuples to lists for JSON serialization
        for k, v in config_dict.items():
            if isinstance(v, tuple):
                config_dict[k] = list(v)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """Load config from json"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
            # Convert lists back to tuples if needed
            for field in fields(cls):
                if str(field.type).startswith('typing.Tuple'):
                    if config_dict.get(field.name):
                        config_dict[field.name] = tuple(config_dict[field.name])
            
            return cls(**config_dict)

    @classmethod
    def from_args(cls: Type[T]) -> T:
        """Create config from command line arguments"""
        parser = argparse.ArgumentParser(description=f'{cls.__name__} Configuration')
        
        # Add arguments for each field in the dataclass
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            default_value = field.default if field.default is not MISSING else field.default_factory()
            
            # Handle special cases
            if str(field_type).startswith('typing.Tuple'):
                parser.add_argument(
                    f'--{field_name}', 
                    nargs='+', 
                    type=int,
                    default=default_value,
                    help=f'{field_name} (tuple of integers)'
                )
            elif str(field_type).startswith('typing.Optional'):
                parser.add_argument(
                    f'--{field_name}',
                    type=str,
                    default=None,
                    help=f'{field_name} (optional string)'
                )
            else:
                parser.add_argument(
                    f'--{field_name}',
                    type=field_type,
                    default=default_value,
                    help=f'{field_name} (type: {field_type})'
                )
        
        # Add config file argument
        parser.add_argument('--config', type=str, help='Path to config file')
        
        args = parser.parse_args()
        
        # If config file is provided, load it and override with CLI arguments
        if args.config:
            config = cls.load(args.config)
            # Update only the arguments that were explicitly set in CLI
            cli_args = {k: v for k, v in vars(args).items() 
                       if k != 'config' and v != parser.get_default(k)}
            for k, v in cli_args.items():
                setattr(config, k, v)
        else:
            # Convert args to dict and create config
            args_dict = vars(args)
            args_dict.pop('config')  # Remove config file argument
            config = cls(**args_dict)
        
        return config

def setup_training_dir(config: Any) -> Path:
    """Setup training directory and save configuration"""
    output_dir = Path(config.output_dir) / config.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "config.json"
    config.save(str(config_path))
    
    # Save code snapshot
    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)
    
    return output_dir


def get_scheduler(optimizer: torch.optim.Optimizer, config: ConfigMixin):
    """Create learning rate scheduler"""
    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_steps - config.warmup_steps,
            eta_min=config.min_lr
        )
    elif config.scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.min_lr / config.learning_rate,
            total_iters=config.num_steps - config.warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    # Wrap with warmup
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=config.warmup_steps
            ),
            scheduler
        ],
        milestones=[config.warmup_steps]
    )
    
    return scheduler

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True