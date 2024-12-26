import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange
from timm.models.vision_transformer import Mlp
import functools
from .dit import PatchEmbed
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        if dim == 1:
            self.dims = [1, 2, 3]
        elif dim == 2:
            self.dims = [1, 2]
        else:
            raise NotImplementedError
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        qkv_bias=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        rotary_freqs = RotaryEmbedding(
            dim=head_dim // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        ).get_axial_freqs(frame_height, frame_width)
        self.register_buffer("rotary_freqs", rotary_freqs, persistent=False)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.frame_height * self.frame_width

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "b (H W) (h d) -> b h H W d", 
                     H=self.frame_height, W=self.frame_width, h=self.num_heads)
        k = rearrange(k, "b (H W) (h d) -> b h H W d", 
                     H=self.frame_height, W=self.frame_width, h=self.num_heads)
        v = rearrange(v, "b (H W) (h d) -> b h H W d", 
                     H=self.frame_height, W=self.frame_width, h=self.num_heads)

        q = apply_rotary_emb(self.rotary_freqs, q)
        k = apply_rotary_emb(self.rotary_freqs, k)

        q = rearrange(q, "b h H W d -> b h (H W) d")
        k = rearrange(k, "b h H W d -> b h (H W) d")
        v = rearrange(v, "b h H W d -> b h (H W) d")

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = rearrange(attn, "b h N d -> b N (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        mlp_ratio=4.0,
        qkv_bias=False,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            frame_height=frame_height,
            frame_width=frame_width,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=dropout,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_height=256,
        input_width=256,
        patch_size=16,
        enc_dim=768,
        enc_depth=6,
        enc_heads=12,
        dec_dim=768,
        dec_depth=6,
        dec_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        use_variational=True,
        **kwargs,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size**2

        self.latent_dim = latent_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            input_height, 
            input_width, 
            patch_size, 
            3, 
            enc_dim,
        )

        # Drop path (stochastic depth) rate for encoder and decoder
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        dec_dpr = [x.item() for x in torch.linspace(0, drop_path, dec_depth)]

        # Encoder
        self.encoder = nn.ModuleList([
            AttentionBlock(
                enc_dim,
                enc_heads,
                self.seq_h,
                self.seq_w,
                mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=enc_dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(enc_depth)
        ])
        self.enc_norm = norm_layer(enc_dim)

        # Bottleneck
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = nn.Linear(enc_dim, mult * latent_dim)
        self.post_quant_conv = nn.Linear(latent_dim, dec_dim)
        
        # Decoder
        self.decoder = nn.ModuleList([
            AttentionBlock(
                dec_dim,
                dec_heads,
                self.seq_h,
                self.seq_w,
                mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dec_dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(dec_depth)
        ])
        self.dec_norm = norm_layer(dec_dim)
        self.predictor = nn.Linear(dec_dim, self.patch_dim)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # patchify
        bsz, _, h, w = x.shape
        x = x.reshape(
            bsz,
            3,
            self.seq_h,
            self.patch_size,
            self.seq_w,
            self.patch_size,
        ).permute([0, 1, 3, 5, 2, 4])  # [b, c, h, p, w, p] --> [b, c, p, p, h, w]
        x = x.reshape(bsz, self.patch_dim, self.seq_h, self.seq_w)  # --> [b, cxpxp, h, w]
        x = x.permute([0, 2, 3, 1]).reshape(bsz, self.seq_len, self.patch_dim)  # --> [b, hxw, cxpxp]
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        # unpatchify
        x = x.reshape(bsz, self.seq_h, self.seq_w, self.patch_dim).permute([0, 3, 1, 2])  # [b, h, w, cxpxp] --> [b, cxpxp, h, w]
        x = x.reshape(
            bsz,
            3,
            self.patch_size,
            self.patch_size,
            self.seq_h,
            self.seq_w,
        ).permute([0, 1, 4, 2, 5, 3])  # [b, c, p, p, h, w] --> [b, c, h, p, w, p]
        x = x.reshape(
            bsz,
            3,
            self.input_height,
            self.input_width,
        )  # [b, c, hxp, wxp]
        return x

    def encode(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)

        # Bottleneck
        moments = self.quant_conv(x)
        if not self.use_variational:
            moments = torch.cat((moments, torch.zeros_like(moments)), 2)
        posterior = DiagonalGaussianDistribution(moments, deterministic=(not self.use_variational), dim=2)
        return posterior

    def decode(self, z):
        # Bottleneck
        z = self.post_quant_conv(z)

        # Decoder
        for blk in self.decoder:
            z = blk(z)
        z = self.dec_norm(z)

        # Predictor
        z = self.predictor(z)

        # Unpatchify
        dec = self.unpatchify(z)
        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if self.use_variational and sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    @torch.no_grad()
    def sample(self, batch_size=1, device="cuda"):
        """
        Sample from the prior distribution and generate images
        """
        # Sample from standard normal distribution
        z = torch.randn(batch_size, self.seq_len, self.latent_dim, device=device)
        # Decode
        samples = self.decode(z)
        return samples

    def get_codebook_indices(self, x):
        """
        Get discrete codebook indices for input images
        """
        posterior = self.encode(x)
        if self.use_variational:
            z = posterior.mode()  # Use mode for deterministic encoding
        else:
            z = posterior.mode()
        # You could add quantization here if needed
        return z

    def decode_from_indices(self, indices):
        """
        Decode from codebook indices
        """
        z = indices  # You could add dequantization here if needed
        dec = self.decode(z)
        return dec
    
from dataclasses import dataclass
from typing import Optional

@dataclass
class VAEConfig:
    latent_dim: int
    patch_size: int
    enc_dim: int
    enc_depth: int
    enc_heads: int
    dec_dim: int
    dec_depth: int
    dec_heads: int
    input_height: int = 360
    input_width: int = 640
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    use_variational: bool = True
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path: float = 0.0

# Model configurations for different scales
MODEL_CONFIGS = {
    # Original shallow encoder configuration
    "vit-l-20-shallow": VAEConfig(
        latent_dim=16,
        patch_size=20,
        enc_dim=1024,
        enc_depth=6,
        enc_heads=16,
        dec_dim=1024,
        dec_depth=12,
        dec_heads=16,
        input_height=180,
        input_width=320,
    ),
    
    # Base configuration (similar to ViT-B/16)
    "vit-b-20": VAEConfig(
        latent_dim=32,
        patch_size=20,
        enc_dim=768,
        enc_depth=12,
        enc_heads=12,
        dec_dim=768,
        dec_depth=12,
        dec_heads=12,
        dropout=0.1,
        attn_dropout=0.1,
        drop_path=0.1,
    ),
    
    # Large configuration (similar to ViT-L/16)
    "vit-l-20": VAEConfig(
        latent_dim=64,
        patch_size=20,
        enc_dim=1024,
        enc_depth=24,
        enc_heads=16,
        dec_dim=1024,
        dec_depth=24,
        dec_heads=16,
        dropout=0.1,
        attn_dropout=0.1,
        drop_path=0.2,
    ),
    
    # Huge configuration (similar to ViT-H/14)
    "vit-h-20": VAEConfig(
        latent_dim=128,
        patch_size=20,
        enc_dim=1280,
        enc_depth=32,
        enc_heads=16,
        dec_dim=1280,
        dec_depth=32,
        dec_heads=16,
        dropout=0.1,
        attn_dropout=0.1,
        drop_path=0.3,
    ),
    
    # Giant configuration (even larger scale)
    "vit-g-20": VAEConfig(
        latent_dim=256,
        patch_size=20,
        enc_dim=1664,
        enc_depth=48,
        enc_heads=16,
        dec_dim=1664,
        dec_depth=48,
        dec_heads=16,
        dropout=0.1,
        attn_dropout=0.1,
        drop_path=0.4,
    ),
}

def create_vae_model(model_size: str, **kwargs) -> "AutoencoderKL":
    """
    Create a VAE model with the specified size and optional overrides.
    
    Args:
        model_size: One of 'vit-b-20', 'vit-l-20', 'vit-h-20', 'vit-g-20'
        **kwargs: Optional parameter overrides
    
    Returns:
        AutoencoderKL model instance
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available sizes: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_size]
    
    # Override config with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return AutoencoderKL(
        latent_dim=config.latent_dim,
        patch_size=config.patch_size,
        enc_dim=config.enc_dim,
        enc_depth=config.enc_depth,
        enc_heads=config.enc_heads,
        dec_dim=config.dec_dim,
        dec_depth=config.dec_depth,
        dec_heads=config.dec_heads,
        input_height=config.input_height,
        input_width=config.input_width,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        use_variational=config.use_variational,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        drop_path=config.drop_path,
    )

# Update the VAE_models dictionary
VAE_models = {
    "vit-l-20-shallow-encoder": lambda **kwargs: create_vae_model("vit-l-20-shallow", **kwargs),
    "vit-b-20": lambda **kwargs: create_vae_model("vit-b-20", **kwargs),
    "vit-l-20": lambda **kwargs: create_vae_model("vit-l-20", **kwargs),
    "vit-h-20": lambda **kwargs: create_vae_model("vit-h-20", **kwargs),
    "vit-g-20": lambda **kwargs: create_vae_model("vit-g-20", **kwargs),
}
