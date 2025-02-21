import torch
import torch.nn.functional as F
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import shutil
from torchvision.utils import make_grid, save_image
from robot_world.utils.train_utils import ConfigMixin, setup_training_dir, get_scheduler, seed_everything
from robot_world.model.vae import create_vae_model, DiagonalGaussianDistribution
from robot_world.data.droid_dataloader import create_droid_dataloader
@dataclass
class VAETrainingConfig(ConfigMixin):
    # Model configuration
    model_type: str = "vit-l-20-shallow"
    seed: int = 42
    
    # Data configuration
    train_batch_size: int = 16
    eval_batch_size: int = 64
    num_workers: int = 4
    image_size: Tuple[int, int] = (360, 640)
    data_dir: str = "/home/kangrui/projects/world_model/dataset"
    dataset_name: str = "droid_100"
    
    # Training configuration
    num_steps: int = 100000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    kld_weight: float = 0.00025
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    scheduler_type: str = "cosine"
    warmup_steps: int = 0
    min_lr: float = 1e-6
    
    # Dataloader configuration
    shuffle_buffer_size: int = 10000
    window_size: int = 1
    future_action_window_size: int = 0
    subsample_length: int = 100
    skip_unlabeled: bool = True
    parallel_calls: int = 16
    traj_transform_threads: int = 16
    traj_read_threads: int = 16
    
    # Logging configuration
    project_name: str = "vae-training"
    exp_name: str = "default"
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 10
    num_viz_samples: int = 32
    output_dir: str = "outputs"
    use_wandb: bool = True
    evaluation_steps:int=100
    
    # Optional resume path
    resume_from: Optional[str] = None

def vae_loss(
    recon_x: torch.Tensor, 
    x: torch.Tensor, 
    posterior: DiagonalGaussianDistribution,
    kld_weight: float = 0.00025
) -> Dict[str, torch.Tensor]:
    """
    Compute VAE loss combining reconstruction loss and KL divergence
    Handles batches with multiple images by computing loss for each and averaging
    """
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

class DualCameraVAETrainer:
    def __init__(self, config: VAETrainingConfig):
        self.config = config
        seed_everything(config.seed)
        
        # Initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            log_with="wandb" if config.use_wandb else None
        )
        
        # Setup directories
        self.output_dir = setup_training_dir(config)
        if self.accelerator.is_main_process:
            self.viz_dir = self.output_dir / "visualizations"
            self.viz_dir.mkdir(exist_ok=True)
            
            # Save training script
            current_file_path = os.path.abspath(__file__)
            script_name = os.path.basename(current_file_path)
            destination_path = os.path.join(self.output_dir / "code", script_name)
            shutil.copy(current_file_path, destination_path)
        
        # Initialize tensorboard if not using wandb
        self.writer = None
        if not config.use_wandb and self.accelerator.is_main_process:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # Create model and optimizer
        self.model = create_vae_model(config.model_type)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Create dataloaders
        self.train_loader = create_droid_dataloader(
            train=True,
            data_dir=config.data_dir,
            batch_size=config.train_batch_size,
            num_workers=config.num_workers,
            shuffle_buffer_size=config.shuffle_buffer_size,
            window_size=config.window_size,
            future_action_window_size=config.future_action_window_size,
            subsample_length=config.subsample_length,
            skip_unlabeled=config.skip_unlabeled,
            resize_primary=config.image_size,
            resize_secondary=config.image_size,
            parallel_calls=config.parallel_calls,
            traj_transform_threads=config.traj_transform_threads,
            traj_read_threads=config.traj_read_threads,
            dataset_names=[config.dataset_name],
            collect_fn=self.custom_collate_fn
        )
        
        self.val_loader = create_droid_dataloader(
            train=False,
            data_dir=config.data_dir,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            shuffle_buffer_size=config.shuffle_buffer_size,
            window_size=config.window_size,
            future_action_window_size=config.future_action_window_size,
            subsample_length=config.subsample_length,
            skip_unlabeled=config.skip_unlabeled,
            resize_primary=config.image_size,
            resize_secondary=config.image_size,
            parallel_calls=config.parallel_calls,
            traj_transform_threads=config.traj_transform_threads,
            traj_read_threads=config.traj_read_threads,
            dataset_names=[config.dataset_name],
            collect_fn=self.custom_collate_fn
        )
        
        # Prepare for distributed training
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        )
        
        # Initialize wandb if requested
        if config.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.project_name,
                config=config.__dict__,
                init_kwargs={"wandb": {"name": config.exp_name}}
            )
    
    def custom_collate_fn(self, batch_list):
        # Initialize empty lists for each camera
        cam1_images_list = []
        cam2_images_list = []
        
        # Extract camera images and convert numpy arrays to tensors
        for sample in batch_list:
            # Convert numpy to tensor and ensure correct dimension order
            cam1 = torch.from_numpy(sample['obs']['camera/image/varied_camera_1_left_image']).float()  # [2, 360, 640, 3]
            cam2 = torch.from_numpy(sample['obs']['camera/image/varied_camera_2_left_image']).float()  # [2, 360, 640, 3]
            
            # Move channels to correct position for pytorch (B, T, H, W, C) -> (B, T, C, H, W)
            cam1 = cam1.permute(0, 3, 1, 2)  # [2, 3, 360, 640]
            cam2 = cam2.permute(0, 3, 1, 2)  # [2, 3, 360, 640]
            
            cam1_images_list.append(cam1)
            cam2_images_list.append(cam2)
        
        # Stack all images from each camera
        cam1_images = torch.stack(cam1_images_list, dim=0)  # [B, 2, 3, 360, 640]
        cam2_images = torch.stack(cam2_images_list, dim=0)  # [B, 2, 3, 360, 640]
        
        # Create simplified batch dictionary with only camera images
        simplified_batch = {
            'obs': {
                'camera/image/varied_camera_1_left_image': cam1_images,
                'camera/image/varied_camera_2_left_image': cam2_images
            }
        }
        
        return simplified_batch

    def process_batch(self, batch):
        """Extract and process images from batch"""
        # Images are already normalized to [0, 1] from robomimic_transform
        # Extract first timestep and ensure channels are in correct position
        cam1_images = batch['obs']['camera/image/varied_camera_1_left_image'][:, 0]  # [B, 3, 360, 640]
        cam2_images = batch['obs']['camera/image/varied_camera_2_left_image'][:, 0]  # [B, 3, 360, 640]
        
        # Stack both camera images along batch dimension
        combined_images = torch.cat([cam1_images, cam2_images], dim=0)  # [2B, 3, 360, 640]
        
        # Convert from [0, 1] to [-1, 1] range for VAE training
        combined_images = 2.0 * combined_images - 1.0
        
        return combined_images
    
    def log_metrics(self, metrics: Dict[str, float], step: int, images: Optional[Dict[str, torch.Tensor]] = None):
        """Log metrics and images to wandb or tensorboard"""
        if self.accelerator.is_main_process:
            if self.config.use_wandb:
                log_dict = metrics.copy()
                if images:
                    log_dict.update({
                        k: wandb.Image(v.cpu().numpy().transpose(1, 2, 0))
                        for k, v in images.items()
                    })
                self.accelerator.log(log_dict, step=step)
            elif self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, step)
                if images:
                    for key, img in images.items():
                        self.writer.add_image(key, img, step)
    
    @torch.no_grad()
    def validate_and_visualize(self, step: int) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Run validation and generate visualizations"""
        self.model.eval()
        val_losses = []
        num_eval_steps = getattr(self.config, 'evaluation_steps', 23)  # Default to 23 if not specified
        
        # Validation loop with tqdm
        with torch.no_grad():
            pbar = tqdm(desc="Validating", total=num_eval_steps, leave=False)
            val_iterator = iter(self.val_loader)
            
            for _ in range(num_eval_steps):
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(self.val_loader)
                    batch = next(val_iterator)
                    
                
                images = self.process_batch(batch)
                recon, posterior = self.model(images)
                losses = vae_loss(recon, images, posterior, self.config.kld_weight)
                val_losses.append({k: v.item() for k, v in losses.items()})
                pbar.update(1)
            
            pbar.close()
            
        if not val_losses:
            return {}, {}
            
        # Calculate average validation losses
        avg_losses = {}
        for k in val_losses[0].keys():
            avg_losses[f'val_{k}'] = sum(x[k] for x in val_losses) / len(val_losses)
        
        # Generate visualizations with visualization-specific iterator
        viz_tensors = {}
        viz_iterator = iter(self.val_loader)
        try:
            viz_batch = next(viz_iterator)
        except StopIteration:
            viz_iterator = iter(self.val_loader)
            viz_batch = next(viz_iterator)
        
        with torch.no_grad():
            images = self.process_batch(viz_batch)
            recon, _ = self.model(images[:self.config.num_viz_samples])
            
            # Convert back to [0, 1] range for visualization
            images = (images + 1.0) / 2.0
            recon = (recon + 1.0) / 2.0
            images = torch.clamp(images, 0, 1)
            recon = torch.clamp(recon, 0, 1)
            
            # Create visualization grids
            input_grid = make_grid(images[:self.config.num_viz_samples], nrow=4, padding=2)
            recon_grid = make_grid(recon[:self.config.num_viz_samples], nrow=4, padding=2)
            
            if self.accelerator.is_main_process:
                save_image(input_grid, self.viz_dir / f'inputs_step_{step}.png')
                save_image(recon_grid, self.viz_dir / f'reconstructions_step_{step}.png')
                
            
            viz_tensors = {
                'inputs': input_grid,
                'reconstructions': recon_grid
            }
        self.model.train()
        return avg_losses, viz_tensors
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state = {
                'model': unwrapped_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': step,
                'config': self.config.__dict__
            }
            
            # Save regular checkpoint
            checkpoint_path = checkpoint_dir / f'checkpoint_{step}.pt'
            self.accelerator.save(state, checkpoint_path)
            
            # Save best checkpoint if requested
            if is_best:
                best_path = checkpoint_dir / 'checkpoint_best.pt'
                self.accelerator.save(state, best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        return checkpoint['step']

    def train(self):
        """Main training loop"""
        start_step = 0
        if self.config.resume_from:
            start_step = self.load_checkpoint(self.config.resume_from)
        
        self.model.train()
        progress_bar = tqdm(
            range(start_step, self.config.num_steps),
            disable=not self.accelerator.is_main_process
        )
        
        train_iter = iter(self.train_loader)
        best_val_loss = float('inf')
        
        for step in progress_bar:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Process batch and run model
            images = self.process_batch(batch)
            with self.accelerator.accumulate(self.model):
                recon, posterior = self.model(images)
                losses = vae_loss(recon, images, posterior, self.config.kld_weight)
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
                gathered_losses = self.accelerator.gather({
                    k: v.detach() for k, v in losses.items()
                })
                
                if self.accelerator.is_main_process:
                    log_dict = {
                        k: v.mean().item() for k, v in gathered_losses.items()
                    }
                    log_dict['learning_rate'] = self.scheduler.get_last_lr()[0]
                    
                    # Run validation and visualization if needed
                    if step % self.config.eval_every == 0:
                        val_metrics, viz_images = self.validate_and_visualize(step)
                        log_dict.update(val_metrics)
                        
                        # Check if best model
                        val_loss = val_metrics['val_loss']
                        is_best = val_loss < best_val_loss
                        if is_best:
                            best_val_loss = val_loss
                            
                        # Log metrics and images
                        self.log_metrics(log_dict, step, viz_images)
                    else:
                        # Log metrics only
                        self.log_metrics(log_dict, step)
                    
                    # Update progress bar
                    progress_bar.set_postfix(**log_dict)
            
            # Save checkpoint
            if step % self.config.save_every == 0:
                self.save_checkpoint(
                    step,
                    is_best=(step % self.config.eval_every == 0 and 
                            val_metrics.get('val_loss', float('inf')) == best_val_loss)
                )
        
        # Final save and cleanup
        if self.accelerator.is_main_process:
            
            self.save_checkpoint(self.config.num_steps, is_best=False)
            
            if self.writer is not None:
        
                self.writer.close()
            
            if self.config.use_wandb:
        
                self.accelerator.end_training()


def main():
    # Parse config from command line arguments
    config = VAETrainingConfig.from_args()
    
    # Initialize trainer
    trainer = DualCameraVAETrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()