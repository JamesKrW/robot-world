from typing import Optional, Dict, Any, Tuple
import torch
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import os
from dataclasses import dataclass
from robot_world.utils.train_utils import ConfigMixin, setup_training_dir, get_scheduler,seed_everything
from robot_world.dataloader.dit_dataloader import create_diffusion_dataloader
import shutil
from robot_world.utils.video_diffusion import VideoDiffusion

@dataclass
class TrainingConfig(ConfigMixin):
    # Model configuration
    dit_model_type: str = "DiT-XS/2"
    vae_model_type: str = "vit-l-20-shallow"
    vae_checkpoint_path: Optional[str] = None
    freeze_vae: bool = True
    
    # Data configuration
    batch_size: int = 32 
    num_workers: int = 4
    image_size: Tuple[int, int] = (360, 640)
    data_dir: str = "/home/kangrui/projects/world_model/droid-debug"
    dataset_name: str = "droid_100"
    
    # Training configuration
    num_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    mixed_precision: str = "fp16"
    
    # Diffusion configuration
    max_noise_level: int = 1000
    
    # Action configuration
    n_action_bins: int = 256
    action_bins_path: Optional[str] = None
    
    # Scheduler configuration
    scheduler_type: str = "cosine"
    warmup_steps: int = 5000
    min_lr: float = 1e-6
    
    # Validation configuration
    validation_freq: int = 1000
    validation_ratio: float = 0.1
    # Visualization parameters
    viz_batch_size: int = 32  # Number of images to show in visualization grid
    num_viz_rows: int = 8    # Number of rows in visualization grid
    
    # Logging configuration
    project_name: str = "dit-training"
    exp_name: str = "default"
    log_every: int = 100
    save_every: int = 1000
    output_dir: str = "outputs"
    use_wandb: bool = True
    
    # Optional resume path
    resume_from: Optional[str] = None
    
    use_gradient_checkpointing: bool = True
    enable_flash_attention: bool = False  # Only works if flash-attn is installed
    gradient_checkpointing_ratio: float = 0.5  # Controls how many layers to checkpoint
    
    seed: int = 42

class DiTTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        # Set seed
        seed_everything(config.seed)
        
        # Setup output directory and save code/config
        self.output_dir = setup_training_dir(config)
        current_file_path = os.path.abspath(__file__)
        script_name = os.path.basename(current_file_path)
        destination_path = os.path.join(self.output_dir / "code", script_name)
        shutil.copy(current_file_path, destination_path)
        
        # Initialize tensorboard writer if wandb is not used
        self.writer = None
        if not config.use_wandb:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # Initialize Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            log_with="wandb" if config.use_wandb else None
        )
        
        # Initialize model
        self.model = VideoDiffusion(
            dit_model_type=config.dit_model_type,
            vae_model_type=config.vae_model_type,
            max_noise_level=config.max_noise_level,
            n_action_bins=config.n_action_bins,
            device=self.accelerator.device
        )
        
    
        # Load VAE checkpoint if provided
        if config.vae_checkpoint_path:
            self.model.load_vae(config.vae_checkpoint_path)
        
        
        # Freeze VAE if specified
        if config.freeze_vae:
            self.model.vae.eval()
            for param in self.model.vae.parameters():
                param.requires_grad = False
        
        # Create optimizer
        trainable_params = (
            self.model.dit.parameters() if config.freeze_vae 
            else list(self.model.dit.parameters()) + list(self.model.vae.parameters())
        )
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Create dataloaders
        self.train_dataloader = create_diffusion_dataloader(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            split='train',
            image_size=config.image_size,
            data_dir=config.data_dir,
            dataset_name=config.dataset_name,
        )
        
        self.val_dataloader = create_diffusion_dataloader(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            split='val',
            image_size=config.image_size,
            data_dir=config.data_dir,
            dataset_name=config.dataset_name,
            validation_ratio=config.validation_ratio,
            image_aug=False,
            shuffle=False
        )
        self.viz_dataloader= create_diffusion_dataloader(
            batch_size=config.viz_batch_size,
            num_workers=config.num_workers,
            split='val',
            image_size=config.image_size,
            data_dir=config.data_dir,
            dataset_name=config.dataset_name,
            validation_ratio=config.validation_ratio,
            image_aug=False,
            shuffle=True
        )
        if config.action_bins_path:
            self.model.load_action_bins(config.action_bins_path)
        else:
            min_vals, max_vals, bin_edges = self.model.action_binner.collect_statistics(self.train_dataloader)
            self.model.action_binner.min_vals = min_vals
            self.model.action_binner.max_vals = max_vals
            self.model.action_binner.bin_edges = bin_edges
            self.model.action_binner.save_bins(os.path.join(self.output_dir, f"bins_{self.model.action_binner.strategy}.json"))
        
        
        # Prepare for distributed training
        (
            self.model.dit,
            self.model.vae,
            self.optimizer,
            self.scheduler,
            self.train_dataloader,
            self.val_dataloader,
        ) = self.accelerator.prepare(
            self.model.dit,
            self.model.vae,
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
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load VAE if it exists in checkpoint
        if 'vae_model' in checkpoint:
            unwrapped_vae = self.accelerator.unwrap_model(self.model.vae)
            unwrapped_vae.load_state_dict(checkpoint['vae_model'])
        
        # Load DiT
        if 'dit_model' in checkpoint:
            unwrapped_dit = self.accelerator.unwrap_model(self.model.dit)
            unwrapped_dit.load_state_dict(checkpoint['dit_model'])
        
        # Load optimizer and scheduler states
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        return checkpoint.get('step', 0)

    def save_checkpoint(self, step: int, is_best: bool = False):
        if self.accelerator.is_main_process:
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f'checkpoint_{step}.pt'
            
            # Get unwrapped models
            unwrapped_dit = self.accelerator.unwrap_model(self.model.dit)
            unwrapped_vae = self.accelerator.unwrap_model(self.model.vae)
            
            state = {
                'dit_model': unwrapped_dit.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': step,
                'config': self.config.__dict__
            }
            
            # Save VAE state only if it's being trained
            if not self.config.freeze_vae:
                state['vae_model'] = unwrapped_vae.state_dict()
            
            self.accelerator.save(state, checkpoint_path)
            
            if is_best:
                best_path = checkpoint_dir / 'checkpoint_best.pt'
                self.accelerator.save(state, best_path)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop and save visualization samples"""
        self.model.dit.eval()
        if not self.config.freeze_vae:
            self.model.vae.eval()
            
        val_losses = []
        print(self.optimizer.state_dict()['state'])
        # Get current step
        # current_step = self.optimizer.state_dict()['state'][0]['step']
        try:
            current_step = self.optimizer.state_dict()['state'][0]['step']
        except KeyError:
            # If no steps have been taken yet, use 0
            current_step = 0
        # visualize samples
        
        self.viz_batch = next(iter(self.viz_dataloader))
        #self.viz_batch = {k: v[:self.config.viz_batch_size] for k, v in self.viz_batch.items()}
            
        # Generate visualization samples
        if self.accelerator.is_main_process:
            image_dir = self.output_dir / "images"
            image_dir.mkdir(exist_ok=True)
            
            # Generate samples with different sampling steps
            for ddim_noise_steps in [10, 50]:  # Try both fast and slow sampling
                sample_path = image_dir / f'val_samples_step{current_step}_ddim{ddim_noise_steps}.png'
                self.model.visualize(
                    self.viz_batch['prev_frames'],
                    self.viz_batch['delta_action'],
                    num_samples=self.config.viz_batch_size,
                    save_path=sample_path,
                    ddim_noise_steps=ddim_noise_steps
                )
            
            # Also visualize the ground truth for reference
            from torchvision.utils import save_image, make_grid
            ground_truth = self.viz_batch['current_frame'][:self.config.viz_batch_size]
            grid = make_grid(ground_truth, nrow=int(self.config.num_viz_rows), padding=2, normalize=False)
            save_image(grid, image_dir / f'val_ground_truth_step{current_step}.png')
        
        # Compute validation loss on the full validation set
        for batch in self.val_dataloader:
            t = torch.randint(
                0, self.config.max_noise_level, (batch['current_frame'].shape[0],),
                device=self.accelerator.device
            )
            
            pred_dict = self.model.process_batch(batch, t)
            loss = self.model.compute_loss(pred_dict)
            val_losses.append(loss.item())
        
        avg_loss = sum(val_losses) / len(val_losses)
        
        self.model.dit.train()
        if not self.config.freeze_vae:
            self.model.vae.train()
            
        return {'val_loss': avg_loss}
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        start_step = 0
        if resume_from:
            start_step = self.load_checkpoint(resume_from)
        
        self.model.dit.train()
        if not self.config.freeze_vae:
            self.model.vae.train()
            
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
            with self.accelerator.accumulate(self.model.dit):
                # Sample random timesteps
                t = torch.randint(
                    0, self.config.max_noise_level, (batch['current_frame'].shape[0],),
                    device=self.accelerator.device
                )
                
                # Process batch and compute loss
                pred_dict = self.model.process_batch(batch, t)
                loss = self.model.compute_loss(pred_dict)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.dit.parameters() if self.config.freeze_vae 
                        else list(self.model.dit.parameters()) + list(self.model.vae.parameters()),
                        1.0
                    )
                    
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            if step % self.config.log_every == 0:
                # Get current learning rate
                current_lr = self.scheduler.get_last_lr()[0]
                
                # Prepare log dict
                log_dict = {
                    'loss': loss.item(),
                    'learning_rate': current_lr
                }
                
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
                
                # Log metrics
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
    trainer = DiTTrainer(config)
    
    # Start/resume training
    trainer.train(resume_from=config.resume_from)

if __name__ == "__main__":
    main()