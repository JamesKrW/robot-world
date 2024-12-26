import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import os
from typing import Optional, Dict, Any
from model.vae_new import AutoencoderKL, DiagonalGaussianDistribution, create_vae_model
from dataloader.vae_dataloader import create_droid_dataloader
from dataclasses import dataclass
from typing import Tuple
from train_utils import ConfigMixin, setup_training_dir

@dataclass
class TrainingConfig(ConfigMixin):
    # Model configuration
    model_type: str = "vit-l-20-shallow"
    
    # Data configuration
    batch_size: int = 32
    num_workers: int = 4
    image_size: Tuple[int, int] = (180, 320)
    data_dir: str = "/home/kangrui/projects/world_model/droid-debug"
    dataset_name: str = "droid_100"
    
    # Training configuration
    num_steps: int = 100000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    kld_weight: float = 0.00025
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    
    # Scheduler configuration
    scheduler_type: str = "cosine"  # "cosine" or "linear"
    warmup_steps: int = 5000
    min_lr: float = 1e-6
    
    # Validation configuration
    validation_ratio: float = 0.1
    validation_freq: int = 1000
    
    # Logging configuration
    project_name: str = "vae-training"
    exp_name: str = "default"
    log_every: int = 100
    save_every: int = 1000
    output_dir: str = "outputs"
    use_wandb: bool = True  # Whether to use WandB logging
    
    # Optional resume path
    resume_from: Optional[str] = None

def get_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
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

def vae_loss(
    recon_x: torch.Tensor, 
    x: torch.Tensor, 
    posterior: DiagonalGaussianDistribution,
    kld_weight: float = 0.00025
) -> Dict[str, torch.Tensor]:
    """Compute VAE loss with KL divergence"""
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    kl_loss = 0.5 * torch.mean(
        posterior.logvar.exp() + posterior.mean.pow(2) - 1.0 - posterior.logvar
    )
    loss = recon_loss + kld_weight * kl_loss
    
    return {
        'loss': loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }

class VAETrainer:
    def __init__(
        self,
        config: TrainingConfig,
    ):
        self.config = config
        
        # Setup output directory and save code/config
        self.output_dir = setup_training_dir(config)
        
        # Initialize tensorboard writer if wandb is not used
        self.writer = None
        if not config.use_wandb and self.accelerator.is_main_process:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # Initialize Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            log_with="wandb" if config.use_wandb else None
        )
        
        # Create model and other components
        self.model = create_vae_model(config.model_type)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Create dataloaders
        self.train_dataloader = create_droid_dataloader(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            split='train',
            image_size=config.image_size,
            data_dir=config.data_dir,
            dataset_name=config.dataset_name,
        )
        
        self.val_dataloader = create_droid_dataloader(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            split='val',
            image_size=config.image_size,
            image_aug=False,
            data_dir=config.data_dir,
            dataset_name=config.dataset_name,
            validation_ratio=config.validation_ratio
        )
        
        # Prepare for distributed training
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataloader,
            self.val_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataloader,
            self.val_dataloader
        )
        
        # Initialize wandb if used
        if config.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.project_name,
                config=config.__dict__,
                init_kwargs={"wandb": {"name": config.exp_name}}
            )
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to either WandB or TensorBoard"""
        if self.accelerator.is_main_process:
            if self.config.use_wandb:
                self.accelerator.log(metrics, step=step)
            elif self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, step)
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_path = checkpoint_dir / f'checkpoint_{step}.pt'
            
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state = {
                'model': unwrapped_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': step,
                'config': self.config.__dict__
            }
            
            self.accelerator.save(state, checkpoint_path)
            
            if is_best:
                best_path = checkpoint_dir / 'checkpoint_best.pt'
                self.accelerator.save(state, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        return checkpoint['step']
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop"""
        self.model.eval()
        val_losses = []
        
        for batch in self.val_dataloader:
            recon, posterior = self.model(batch)
            losses = vae_loss(recon, batch, posterior, self.config.kld_weight)
            val_losses.append({k: v.item() for k, v in losses.items()})
        
        avg_losses = {}
        for k in val_losses[0].keys():
            avg_losses[f'val_{k}'] = sum(x[k] for x in val_losses) / len(val_losses)
        
        self.model.train()
        return avg_losses
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        start_step = 0
        if resume_from:
            start_step = self.load_checkpoint(resume_from)
        
        self.model.train()
        progress_bar = tqdm(
            range(start_step, self.config.num_steps),
            disable=not self.accelerator.is_main_process
        )
        
        train_iter = iter(self.train_dataloader)
        best_val_loss = float('inf')
        
        for step in progress_bar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            
            # Forward pass
            with self.accelerator.accumulate(self.model):
                recon, posterior = self.model(batch)
                losses = vae_loss(recon, batch, posterior, self.config.kld_weight)
                loss = losses['loss']
                
                # Backward pass
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            if step % self.config.log_every == 0:
                # Gather losses from all processes
                gathered_losses = self.accelerator.gather(
                    {k: v.detach() for k, v in losses.items()}
                )
                
                # Log only from main process
                if self.accelerator.is_main_process:
                    # Get current learning rate
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    # Prepare log dict
                    log_dict = {
                        k: v.mean().item() 
                        for k, v in gathered_losses.items()
                    }
                    log_dict['learning_rate'] = current_lr
                    
                    # Run validation if needed
                    if step % self.config.validation_freq == 0:
                        val_metrics = self.validate()
                        log_dict.update(val_metrics)
                        
                        # Check if best model
                        val_loss = val_metrics['val_loss']
                        is_best = val_loss < best_val_loss
                        if is_best:
                            best_val_loss = val_loss
                    
                    # Update progress bar
                    progress_bar.set_postfix(**log_dict)
                    
                    # Log metrics using the appropriate logger
                    self.log_metrics(log_dict, step)
            
            # Save checkpoint
            if step % self.config.save_every == 0:
                self.save_checkpoint(
                    step,
                    is_best=(step % self.config.validation_freq == 0 and 
                            'val_loss' in log_dict and 
                            log_dict['val_loss'] == best_val_loss)
                )
        
        # Final save and cleanup
        if self.accelerator.is_main_process:
            self.save_checkpoint(self.config.num_steps, is_best=False)
            
            # Close tensorboard writer if it exists
            if self.writer is not None:
                self.writer.close()
            
            # End wandb if it's being used
            if self.config.use_wandb:
                self.accelerator.end_training()

def main():
    # Get config from CLI arguments
    config = TrainingConfig.from_args()
    
    # Initialize trainer
    trainer = VAETrainer(config)
    
    # Start/resume training
    trainer.train(resume_from=config.resume_from)

if __name__ == "__main__":
    main()