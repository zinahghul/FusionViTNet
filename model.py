import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os 
import math

class CarotidDataset(Dataset):
    def __init__(self, us_images_dir, mask_images_dir, transform=None):
        self.us_images = sorted([os.path.join(us_images_dir, fname) for fname in os.listdir(us_images_dir)])
        self.mask_images = sorted([os.path.join(mask_images_dir, fname) for fname in os.listdir(mask_images_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.us_images)

    def __getitem__(self, idx):
        us_image = Image.open(self.us_images[idx]).convert('L')
        mask = Image.open(self.mask_images[idx]).convert('L')

        if self.transform:
            us_image = self.transform(us_image)
            mask = self.transform(mask)

        return us_image, mask


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.depthwise(x))
        x = self.pointwise(x)
        return x


# Define UNetBaseline Class first
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleUNet, self).__init__()

        # Encoder
        self.encoder1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)

        # Final output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        e1 = F.relu(self.encoder1(x))
        e2 = F.relu(self.encoder2(self.pool(e1)))

        # Decoding path
        up1 = self.upconv1(e2)
        up1 = torch.cat([up1, e1], dim=1)  # Concatenate skip connections
        up2 = self.upconv2(up1)

        out = self.final_conv(up2)
        return out

class SimpleViT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = 32
        self.patch_embed = nn.Conv2d(in_channels, self.embed_dim, 
                                    kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=2,
            dim_feedforward=64
        )
        self.reconstruct = nn.ConvTranspose2d(
            self.embed_dim, out_channels,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.reconstruct(x)
        return x

class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, num_patches, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 1, 3)
        x = x.view(B, num_patches, -1)
        return x

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim, patch_size):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * patch_size * 1, projection_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, projection_dim))
        
    def forward(self, x):
        x = self.projection(x)
        x = x + self.position_embedding
        return x

class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim)
        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=0.1)
        self.norm2 = nn.LayerNorm(projection_dim)
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * mlp_ratio, projection_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class FusionViTNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, image_size=256):
        super().__init__()
        self.patch_size = 16
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_layers = 8
        
        assert image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // self.patch_size) ** 2
        
        # CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            DepthwiseSeparableConv(16, 32, stride=2),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 32),
            nn.ReLU()
        )
        
        # ViT components
        self.patches = Patches(self.patch_size)
        self.patch_encoder = PatchEncoder(
            num_patches=self.num_patches,
            projection_dim=self.projection_dim,
            patch_size=self.patch_size
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.projection_dim, self.num_heads) 
            for _ in range(self.transformer_layers)
        ])
        self.vit_norm = nn.LayerNorm(self.projection_dim)
        
        # Fusion components
        self.fusion_conv = nn.Conv2d(32 + self.projection_dim, 32, kernel_size=1)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == 256, f"Input size must be 256x256, got {H}x{W}"
        
        # CNN pathway
        cnn_features = self.encoder(x)
        
        # ViT pathway
        patches = self.patches(x)
        encoded_patches = self.patch_encoder(patches)
        
        for block in self.transformer_blocks:
            encoded_patches = block(encoded_patches)
        vit_features = self.vit_norm(encoded_patches)
        
        h = w = int(math.sqrt(self.num_patches))
        vit_features = vit_features.transpose(1, 2).view(B, self.projection_dim, h, w)
        vit_features = F.interpolate(vit_features, size=cnn_features.shape[2:], mode='bilinear')
        
        fused_features = torch.cat([cnn_features, vit_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        out = self.decoder(fused_features)
        return out

# Define ConvEncoderBlock used in the HybridViTUNet (FusionViT)
class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvEncoderBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        return x