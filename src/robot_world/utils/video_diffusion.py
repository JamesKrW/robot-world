from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from torch import autocast
import os
from robot_world.model.dit import DiT_models
from robot_world.model.vae import create_vae_model
from robot_world.utils.action_binner import ActionBinner
from torchvision.utils import make_grid, save_image
from einops import rearrange

class VideoDiffusion:
    def __init__(
        self,
        dit_model_type: str = "DiT-S/2",
        vae_model_type: str = "vit-l-20-shallow",
        max_noise_level: int = 1000,
        n_action_bins: int = 256,
        stabilization_level: int = 15,
        device: str = "cuda"
    ):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.stabilization_level = stabilization_level
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
        # print(f"Loading VAE from {checkpoint_path}")
        # checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # if 'vae_model' in checkpoint:
        #     self.vae.load_state_dict(checkpoint['vae_model'])
        # else:
        #     self.vae.load_state_dict(checkpoint)
        print(f"Loading VAE from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:  # Handle VAE checkpoint format
            unwrapped = {k.replace('model.', ''): v for k, v in checkpoint['model'].items()}
            self.vae.load_state_dict(unwrapped)
        else:
            self.vae.load_state_dict(checkpoint['vae_model'])
    
    def load_action_bins(self, bin_path: str):
        """Load action bins."""
        self.action_binner.load_bins(bin_path)
    
    def process_batch(self, batch: Dict[str, torch.Tensor], t: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract inputs
        prev_frames = batch['prev_frames'].to(self.device)  # [B, T, C, H, W]
        current_frame = batch['current_frame'].to(self.device)  # [B, C, H, W]
        delta_action = batch['delta_action'].to(self.device)
        
        B, T, C, H, W = prev_frames.shape
        scaling_factor = 0.07843137255
        
        # Convert delta_action to one-hot
        one_hot_action = self.action_binner.convert_to_onehot(delta_action)
        
        # Encode frames with VAE
        with torch.no_grad():
            # Encode prev_frames
            prev_frames = rearrange(prev_frames, "b t c h w -> (b t) c h w")
            prev_latents = self.vae.encode(prev_frames * 2 - 1).mean * scaling_factor
            prev_latents = rearrange(prev_latents, "(b t) (h w) c -> b t c h w",
                                t=T, h=H // self.vae.patch_size, w=W // self.vae.patch_size)
            
            # Encode current frame
            x_0 = self.vae.encode(current_frame * 2 - 1).mean * scaling_factor
            x_0 = rearrange(x_0, "b (h w) c -> b 1 c h w",  # Add time dimension
                        h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        
        # Sample noise and diffuse
        eps = torch.randn_like(x_0)
        alpha_bar = self.alphas_cumprod[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        x_t = sqrt_alpha_bar.view(-1, 1, 1, 1, 1) * x_0 + sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1) * eps
        
        # Combine latents
        all_latents = torch.cat([prev_latents, x_t], dim=1)  # [B, T+1, C, H, W]
        
        # Timesteps - similar to inference
        t_context = torch.full((B, T), self.stabilization_level, device=self.device)
        t_expanded = torch.cat([t_context, t.unsqueeze(1)], dim=1)  # [B, T+1]
        
        # Expand action condition for all frames including target
        one_hot_action = one_hot_action.unsqueeze(1).expand(-1, T+1, -1)  # [B, T+1, action_dim]
        
        # Get prediction for the entire sequence
        eps_theta = self.dit(all_latents, t_expanded, one_hot_action)  # [B, T+1, C, H, W]
        
        # Only return the prediction for the target frame
        eps_theta = eps_theta[:, -1:]  # [B, 1, C, H, W]
        
        return {
            'eps_theta': eps_theta,
            'eps': eps,
            'x_0': x_0,
            'x_t': x_t,
            't': t
        }

    @torch.no_grad()
    def ddim_sample(
        self,
        prev_frames: torch.Tensor,  # [B, T, C, H, W]
        delta_action: torch.Tensor,
        ddim_noise_steps: int = 10,
        eta: float = 0.0,
        noise_abs_max = 20,
        scaling_factor = 0.07843137255,
        stabilization_level = 15
    ) -> torch.Tensor:
        # Move inputs to device first
        prev_frames = prev_frames.to(self.device)
        delta_action = delta_action.to(self.device)
        
        B, T, C, H, W = prev_frames.shape
        noise_range = torch.linspace(-1, self.max_noise_level - 1, ddim_noise_steps + 1).to(self.device)
        
        # VAE encode context frames
        with torch.no_grad():
            prev_frames = rearrange(prev_frames, "b t c h w -> (b t) c h w")
            prev_latents = self.vae.encode(prev_frames * 2 - 1).mean * scaling_factor
            # print(prev_latents.shape)
            prev_latents = rearrange(prev_latents, "(b t) (h w) c -> b t c h w",
                                    t=T, h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        
        # Prepare action conditioning
        one_hot_action = self.action_binner.convert_to_onehot(delta_action)
        T_total = T + 1  # Total sequence length including target frame
        one_hot_action = one_hot_action.unsqueeze(1).expand(-1, T_total, -1)  # [B, T+1, action_dim]
        
        # Initialize noise for target frame with time dimension
        
        chunk = torch.randn((B, 1, *prev_latents.shape[-3:]), device=self.device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x= torch.cat([prev_latents, chunk], dim=1)
        start_frame = max(0, T_total - self.dit.max_frames)
        alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")
        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            # set up noise values
            t_ctx = torch.full((B, T), stabilization_level - 1, dtype=torch.long, device=self.device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=self.device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=self.device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # sliding window
            x_curr = x.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    #print(x_curr.shape,t.shape,one_hot_action[:, start_frame :].shape)
                    v = self.dit(x_curr, t, one_hot_action[:, start_frame :])

            x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

            # get frame prediction
            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]
        
        # Decode final frame
        T= x.shape[1]
        x = rearrange(x, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            x = (self.vae.decode(x / scaling_factor) + 1) / 2
        x = rearrange(x, "(b t) c h w -> b t h w c", t=T)
        pred_images= x[:, -1]
        pred_images=torch.clamp(pred_images, 0, 1)
        return pred_images


    @torch.no_grad()
    def visualize(
        self,
        prev_frames: torch.Tensor,
        delta_action: torch.Tensor,
        num_samples: int = 8,
        save_path: Optional[str] = None,
        ddim_noise_steps: int = 10
    ) -> torch.Tensor:
        """Generate and visualize multiple predictions."""
        # Generate predictions
        pred_images = self.ddim_sample(prev_frames[:num_samples], delta_action[:num_samples], ddim_noise_steps=ddim_noise_steps)
        pred_images = rearrange(pred_images, "b h w c-> b c h w")
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