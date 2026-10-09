"""Vision Transformer (ViT) with Multi-Head Self-Attention extraction and patch token inspection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention (MHSA) module with attention matrix retention."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.last_attn_weights = None  # [B, num_heads, num_tokens, num_tokens]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, num_heads, N, head_dim]

        # Scaled Dot-Product Attention: Softmax(Q K^T / sqrt(d_k))
        scale = 1.0 / np.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        self.last_attn_weights = attn.detach()

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer Encoder block with LayerNorm, MHSA, and MLP with residual skips."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniVisionTransformer(nn.Module):
    """Compact Vision Transformer designed for 28x28 digit recognition with live patch attention telemetry.

    Divides 28x28 images into 16 patches of 7x7 pixels.
    Total tokens: 16 patch tokens + 1 [CLS] token = 17 tokens.
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_chans: int = 1,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 16 patches (4x4 grid)
        self.patch_dim = in_chans * patch_size * patch_size  # 49 pixels per patch
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear Patch Projection
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # Learnable Class Token and 1D Position Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head (on [CLS] token)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts patch tokens, adds [CLS] and position embedding, and passes through blocks."""
        B = x.shape[0]
        # Patchify 28x28 -> [B, 16, 49]
        p = self.patch_size
        # Unfold into patches: (B, 1, 4, 7, 4, 7) -> (B, 16, 49)
        patches = x.unfold(2, p, p).unfold(3, p, p)  # [B, 1, 4, 4, 7, 7]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, self.num_patches, self.patch_dim)

        x_embed = self.patch_embed(patches)  # [B, 16, embed_dim]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, x_embed), dim=1)  # [B, 17, embed_dim]
        x_seq = x_seq + self.pos_embed

        for block in self.blocks:
            x_seq = block(x_seq)

        x_norm = self.norm(x_seq)
        return x_norm, patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self.forward_features(x)
        cls_out = feat[:, 0]
        return self.head(cls_out)

    def forward_with_attention(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Performs forward pass and captures full attention matrices across all heads."""
        self.eval()
        with torch.no_grad():
            feat, patches = self.forward_features(x)
            cls_out = feat[:, 0]
            logits = self.head(cls_out)
            probs = F.softmax(logits, dim=-1)

            # Last block attention matrix: [1, num_heads, 17, 17]
            attn_weights = self.blocks[-1].attn.last_attn_weights[0].cpu().numpy()

        return {
            "logits": logits[0].cpu().numpy(),
            "probabilities": probs[0].cpu().numpy(),
            "attention_weights": attn_weights,  # [num_heads, 17, 17]
            "cls_attention": attn_weights[:, 0, 1:],  # [num_heads, 16] - CLS token attention to each patch
            "patches": patches[0].cpu().numpy(),  # [16, 49]
            "input": x[0, 0].cpu().numpy()
        }
