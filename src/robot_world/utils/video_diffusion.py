from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from torch import autocast
import os
from robot_world.model.dit import DiT_models
from robot_world.model.vae import create_vae_model
from robot_world.utils.action_binner import ActionBinner
from torchvision.utils import make_grid, save_image

class VideoDiffusion:
    def __init__(
        self,
        dit_model_type: str = "DiT-S/2",
        vae_model_type: str = "vit-l-20-shallow",
        max_noise_level: int = 1000,
        n_action_bins: int = 256,
        device: str = "cuda"
    ):
        self.device = device
        self.max_noise_level = max_noise_level
        
        # Initialize models
        self.dit = DiT_models[dit_model_type]().to(device)
        self.vae = create_vae_model(vae_model_type).to(device)
        
        # Initialize action binner
        self.action_binner = ActionBinner(n_bins=n_action_bins)
        
        # Setup diffusion parameters
        self.betas = self._sigmoid_beta_schedule(max_noise_level)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
    
    def _sigmoid_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """Generate the beta schedule for diffusion."""
        beta_start = 1e-4
        beta_end = 2e-2
        betas = torch.sigmoid(torch.linspace(-10, 10, timesteps)) * (beta_end - beta_start) + beta_start
        return betas

    def load_dit(self, checkpoint_path: str):
        """Load DiT checkpoint."""
        print(f"Loading DiT from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'dit_model' in checkpoint:
            self.dit.load_state_dict(checkpoint['dit_model'])
        else:
            self.dit.load_state_dict(checkpoint)
    
    def load_vae(self, checkpoint_path: str):
        """Load VAE checkpoint."""
        print(f"Loading VAE from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'vae_model' in checkpoint:
            self.vae.load_state_dict(checkpoint['vae_model'])
        else:
            self.vae.load_state_dict(checkpoint)
    
    def load_action_bins(self, bin_path: str):
        """Load action bins."""
        self.action_binner.load_bins(bin_path)
    
    def process_batch(self, batch: Dict[str, torch.Tensor], t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a batch of data for training/validation
        
        Args:
            batch: Dictionary containing 'prev_frames', 'current_frame', and 'delta_action'
            t: Timesteps tensor of shape (batch_size,)
            
        Returns:
            Dictionary containing model outputs and targets for loss computation
        """
        # Extract inputs
        prev_frames = batch['prev_frames'].to(self.device)
        current_frame = batch['current_frame'].to(self.device)
        delta_action = batch['delta_action'].to(self.device)
        
        # Convert delta_action to one-hot
        one_hot_action = self.action_binner.convert_to_onehot(delta_action)
        
        # Encode frames with VAE
        B = prev_frames.shape[0]
        scaling_factor = 0.07843137255
        
        with torch.no_grad():
            prev_latents = self.vae.encode(prev_frames * 2 - 1).mean * scaling_factor
            x_0 = self.vae.encode(current_frame * 2 - 1).mean * scaling_factor
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Get diffusion parameters for the timesteps
        alpha_bar = self.alphas_cumprod[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        
        # Forward diffusion process
        # q(x_t | x_0) = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
        x_t = sqrt_alpha_bar.view(-1, 1, 1, 1) * x_0 + sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1) * eps
        
        # Combine prev_latents and noisy latent for temporal context
        all_latents = torch.cat([prev_latents, x_t.unsqueeze(1)], dim=1)
        
        # Expand timesteps for all frames (keeping context frames at stabilization level)
        t_context = torch.full((B, prev_latents.size(1)), 15, device=self.device)
        t_expanded = torch.cat([t_context, t.unsqueeze(1)], dim=1)
        
        # Get model prediction
        eps_theta = self.dit(all_latents, t_expanded, one_hot_action)
        
        return {
            'eps_theta': eps_theta,    # Model's noise prediction
            'eps': eps,                # Target noise
            'x_0': x_0,               # Original clean sample
            'x_t': x_t,               # Noised sample
            't': t                     # Timesteps
        }

    @torch.no_grad()
    def ddim_sample(
        self,
        prev_frames: torch.Tensor,
        delta_action: torch.Tensor,
        num_steps: int = 10,
        eta: float = 0.0  # η=0 for deterministic sampling
    ) -> torch.Tensor:
        """Generate next frame using DDIM sampling."""
        B = prev_frames.shape[0]
        device = self.device
        noise_range = torch.linspace(-1, self.max_noise_level - 1, num_steps + 1).to(device)
        
        # VAE encode context frames
        scaling_factor = 0.07843137255
        prev_latents = self.vae.encode(prev_frames * 2 - 1).mean * scaling_factor
        
        # Convert action to one-hot
        one_hot_action = self.action_binner.convert_to_onehot(delta_action)
        
        # Initialize with random noise
        x = torch.randn_like(prev_latents[:, 0])
        x = torch.clamp(x, -20, 20)  # Stabilize initial noise
        
        # DDIM sampling loop
        for noise_idx in reversed(range(1, len(noise_range))):
            # Set up noise values
            t = torch.full((B,), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((B,), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            
            # Get model prediction
            all_latents = torch.cat([prev_latents, x.unsqueeze(1)], dim=1)
            t_context = torch.full((B, prev_latents.size(1)), 15, device=device)
            t_expanded = torch.cat([t_context, t.unsqueeze(1)], dim=1)
            
            with autocast(device, dtype=torch.float16):
                eps_theta = self.dit(all_latents, t_expanded, one_hot_action)
            
            # DDIM update step
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]
            
            # Predict x_0
            c1 = torch.sqrt(1 / alpha)
            c2 = torch.sqrt(1 - alpha) - eta * torch.sqrt(alpha)
            x_0_pred = c1 * x - c2 * eps_theta
            
            # Update x_t
            c3 = torch.sqrt(alpha_next)
            c4 = torch.sqrt(1 - alpha_next) - eta * torch.sqrt(alpha_next)
            x = c3 * x_0_pred + c4 * eps_theta
        
        # Decode final prediction
        pred_image = self.vae.decode(x / scaling_factor)
        pred_image = (pred_image + 1) / 2  # Scale to [0, 1]
        pred_image = torch.clamp(pred_image, 0, 1)
        
        return pred_image

    @torch.no_grad()
    def visualize(
        self,
        prev_frames: torch.Tensor,
        delta_action: torch.Tensor,
        num_samples: int = 8,
        save_path: Optional[str] = None,
        num_steps: int = 10
    ) -> torch.Tensor:
        """Generate and visualize multiple predictions."""
        # Generate predictions
        pred_images = self.ddim_sample(prev_frames[:num_samples], delta_action[:num_samples], num_steps=num_steps)
        
        # Create grid
        grid = make_grid(pred_images, nrow=int(num_samples ** 0.5), padding=2, normalize=False)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(grid, save_path)
        
        return grid

    def compute_loss(self, pred_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the DDPM training loss."""
        return F.mse_loss(pred_dict['eps_theta'], pred_dict['eps'])