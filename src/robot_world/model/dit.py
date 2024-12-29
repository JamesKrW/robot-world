"""
References:
    
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/unet3d.py
    - Latte: https://github.com/Vchitect/Latte/blob/main/models/latte.py
"""

from typing import Optional, Literal
import torch
from torch import nn
from robot_world.model.rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from robot_world.model.attention import SpatialAxialAttention, TemporalAxialAttention
from timm.models.vision_transformer import Mlp
from timm.layers.helpers import to_2tuple
import math


def modulate(x, shift, scale):
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // scale.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def gate(x, g):
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_height=256,
        img_width=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        assert random_sample or (H == self.img_size[0] and W == self.img_size[1]), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C H W -> B (H W) C")
        else:
            x = rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),  # hidden_size is diffusion model hidden size
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_attn = SpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_attn = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        B, T, H, W, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=25,
        max_frames=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads)
        self.external_cond = nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, external_cond=None):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        """

        B, T, C, H, W = x.shape

        # add spatial embeddings
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)  # (B*T, C, H, W) -> (B*T, H/2, W/2, D) , C = 16, D = d_model
        # restore shape
        x = rearrange(x, "(b t) h w d -> b t h w d", t=T)
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(t)  # (N, D)
        c = rearrange(c, "(b t) d -> b t d", t=T)
        if torch.is_tensor(external_cond):
            c += self.external_cond(external_cond)
        for block in self.blocks:
            x = block(x, c)  # (N, T, H, W, D)
        x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x


def DiT_XS_2():
    return DiT(
        patch_size=2,
        hidden_size=512,    # Smaller hidden size
        depth=12,           # Fewer layers
        num_heads=8,        # Fewer attention heads
    )

def DiT_S_2():             # Current size
    return DiT(
        patch_size=2,
        hidden_size=1024,
        depth=16,
        num_heads=16,
    )

def DiT_B_2():             # Bigger
    return DiT(
        patch_size=2,
        hidden_size=1536,   # 1.5x hidden size
        depth=24,           # 1.5x depth
        num_heads=24,       # More heads
    )

def DiT_L_2():             # Large
    return DiT(
        patch_size=2,
        hidden_size=2048,   # 2x hidden size
        depth=32,           # 2x depth
        num_heads=32,       # 2x heads
    )

def DiT_XL_2():            # Extra Large
    return DiT(
        patch_size=2,
        hidden_size=2816,   # ~2.75x hidden size
        depth=40,           # 2.5x depth
        num_heads=44,       # More heads for larger hidden size
    )

# Modified DiT models dictionary
DiT_models = {
    "DiT-XS/2": DiT_XS_2,
    "DiT-S/2": DiT_S_2,    # Original size
    "DiT-B/2": DiT_B_2,
    "DiT-L/2": DiT_L_2,
    "DiT-XL/2": DiT_XL_2,
}



if __name__ == "__main__":
    import torch
    from collections import defaultdict
    import pandas as pd
    from typing import Dict, Any
    import numpy as np
    import argparse
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