import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvNxN(in_, out, kernel_size=3, dilation=1):
    """Flexible convolution with configurable kernel size and dilation"""
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
    return nn.Conv2d(in_, out, kernel_size, padding=padding, dilation=dilation)

class Down(nn.Module):
    """U-Net encoder block with standard max pooling"""
    def __init__(self, in_ch, out_ch, kernel_size=3, downsample=True, dilation=1):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2, stride=2)  # No indices needed
        
        self.conv = nn.Sequential(
            ConvNxN(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            ConvNxN(out_ch, out_ch, kernel_size=3, dilation=dilation),  # Apply dilation to both convs
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        skip = x  # Save for skip connection before pooling
        if self.downsample:
            x = self.pool(x)  # Simple pooling, no indices
        x = self.conv(x)
        if self.downsample:
            return x, skip  # Only return features and skip
        return x

class CoarseDown(nn.Module):
    """Coarse stream encoder with stride=2 downsampling and dilated convolutions"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, downsample_factor=2):
        super().__init__()
        
        # Use stride=2 convolution for learnable downsampling (better information preservation)
        self.downsample_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=downsample_factor, padding=1)
        
        self.conv = nn.Sequential(
            ConvNxN(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            ConvNxN(out_ch, out_ch, kernel_size=3, dilation=dilation),  # Apply dilation to both convs
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        skip = x  # Save for skip connection before downsampling
        
        # Apply learnable 2x downsampling with stride=2
        x = self.downsample_conv(x)
        
        # Apply dilated convolutions
        x = self.conv(x)
        
        return x, skip

class Up(nn.Module):
    """Original U-Net decoder block with bilinear upsampling"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        
        # Use bilinear upsampling (standard U-Net approach)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Skip connections have half the channels of the decoder input
            # After concatenation: in_ch (upsampled) + in_ch//2 (skip) = in_ch + in_ch//2
            self.conv = nn.Sequential(
                ConvNxN(in_ch + in_ch//2, out_ch, kernel_size=3),  # Handle concatenated channels
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                ConvNxN(out_ch, out_ch, kernel_size=3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        else:
            # Alternative: transpose convolution
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                ConvNxN(in_ch, out_ch, kernel_size=3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                ConvNxN(out_ch, out_ch, kernel_size=3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

    def forward(self, x, skip):
        # Upsample the input
        x = self.up(x)
        
        # Handle dimension mismatches (standard U-Net approach)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection with upsampled features
        x = torch.cat([skip, x], dim=1)
        
        # Apply double convolution
        return self.conv(x)

class FeaturePyramidAttention(nn.Module):
    """Feature Pyramid Attention (FPA) Module from PANet paper
    Fuses multi-scale features using attention mechanisms
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # 1x1 convolutions to unify channel dimensions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        
        # Global Average Pooling branch
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention generation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final output convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps at different scales [finest -> coarsest]
        """
        # Get target size from finest feature map
        target_size = features[0].shape[2:]
        
        # Apply lateral convolutions and resize to target size
        lateral_features = []
        for i, feature in enumerate(features):
            lateral = self.lateral_convs[i](feature)
            if lateral.shape[2:] != target_size:
                lateral = F.interpolate(lateral, size=target_size, mode='bilinear', align_corners=False)
            lateral_features.append(lateral)
        
        # Sum all lateral features
        fused_feature = sum(lateral_features)
        
        # Global context branch
        gap_feature = self.gap_branch(fused_feature)
        gap_feature = F.interpolate(gap_feature, size=target_size, mode='bilinear', align_corners=False)
        
        # Add global context
        enhanced_feature = fused_feature + gap_feature
        
        # Generate attention weights
        attention = self.attention_conv(enhanced_feature)
        
        # Apply attention to fused feature
        attended_feature = fused_feature * torch.sigmoid(attention)
        
        # Final output
        output = self.output_conv(attended_feature)
        
        return output

class GlobalAttentionUpsample(nn.Module):
    """Global Attention Upsample (GAU) Module from PANet paper
    Uses global context to guide feature upsampling with attention
    """
    def __init__(self, low_in_channels, high_in_channels, out_channels, upsample_scale=2):
        super().__init__()
        self.upsample_scale = upsample_scale
        
        # Low-level feature processing
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # High-level feature processing
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling for attention generation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Attention generation network
        self.attention_fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, low_feature, high_feature):
        """
        Args:
            low_feature: Lower-level (higher resolution) feature
            high_feature: Higher-level (lower resolution) feature with semantic info
        """
        # Process low-level features
        low_feat = self.low_conv(low_feature)
        
        # Process and upsample high-level features
        high_feat = self.high_conv(high_feature)
        if self.upsample_scale > 1:
            high_feat = F.interpolate(high_feat, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
        
        # Ensure spatial dimensions match
        if high_feat.shape[2:] != low_feat.shape[2:]:
            high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Generate channel attention from high-level features
        B, C, H, W = high_feat.shape
        gap = self.global_pool(high_feat).view(B, C)
        channel_attention = self.attention_fc(gap).view(B, C, 1, 1)
        
        # Apply channel attention to low-level features
        low_feat_attended = low_feat * channel_attention
        
        # Generate spatial attention
        spatial_att = self.spatial_attention(high_feat)
        
        # Apply spatial attention
        low_feat_final = low_feat_attended * spatial_att
        
        # Fuse features
        fused = low_feat_final + high_feat
        output = self.fusion_conv(fused)
        
        return output

class PANetFusion(nn.Module):
    """PANet-style fusion for dual-stream architecture
    Implements Feature Pyramid Attention for multi-scale fusion
    """
    def __init__(self):
        super().__init__()
        
        # Feature Pyramid Attention for coarse and fine features
        self.fpa = FeaturePyramidAttention(
            in_channels_list=[1024, 1024],  # fine_features, coarse_features
            out_channels=1024
        )
        
        # Global Attention Upsample for final fusion
        self.gau = GlobalAttentionUpsample(
            low_in_channels=1024,    # fine features (detailed)
            high_in_channels=1024,   # coarse features (semantic)
            out_channels=1024,
            upsample_scale=1  # Both features are same resolution after interpolation
        )
        
    def forward(self, fine_features, coarse_features):
        """
        Args:
            fine_features: [B, 1024, H, W] - Fine-scale features with details
            coarse_features: [B, 1024, H, W] - Coarse-scale features with context (upsampled)
        """
        # Apply Feature Pyramid Attention
        # Treat fine as lower-level (more detailed) and coarse as higher-level (more semantic)
        pyramid_features = [fine_features, coarse_features]
        fpa_output = self.fpa(pyramid_features)
        
        # Apply Global Attention Upsample
        # Use coarse features to generate attention for fine features
        gau_output = self.gau(fine_features, coarse_features)
        
        # Combine FPA and GAU outputs
        final_output = (fpa_output + gau_output) / 2
        
        return final_output

class HNet(nn.Module):
    """Multi-Scale Spatial HNet with PANet fusion"""
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Initial convolution blocks (like original U-Net)
        self.fine_inc = nn.Sequential(
            ConvNxN(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvNxN(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Coarse stream: 2x downsampling with large kernel for initial context
        self.coarse_inc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Learnable 2x downsample input
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvNxN(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Fine-scale encoder (standard 3x3 kernels for local details)
        self.fine_enc1 = Down(64, 128, kernel_size=3)   # Local patterns  
        self.fine_enc2 = Down(128, 256, kernel_size=3)  # Crack textures
        self.fine_enc3 = Down(256, 512, kernel_size=3)  # Fine features
        self.fine_enc4 = Down(512, 1024, kernel_size=3) # Deep features
        
        # Coarse-scale encoder with gradual dilation increase and stride=2 downsampling
        self.coarse_enc1 = CoarseDown(64, 128, kernel_size=5, dilation=1)   # Standard dilation at shallow level
        self.coarse_enc2 = CoarseDown(128, 256, kernel_size=5, dilation=2)  # Start expanding receptive field
        self.coarse_enc3 = CoarseDown(256, 512, kernel_size=5, dilation=4)  # Larger context capture
        self.coarse_enc4 = CoarseDown(512, 1024, kernel_size=5, dilation=8) # Maximum context understanding

        # Bottleneck processing (symmetric design)
        self.fine_bottleneck = nn.Sequential(
            ConvNxN(1024, 1024, kernel_size=3),
            nn.ReLU()
        )
        # Coarse stream: same depth as fine stream with final dilation
        self.coarse_bottleneck = nn.Sequential(
            ConvNxN(1024, 1024, kernel_size=3, dilation=4),  # Keep large receptive field at bottleneck
            nn.ReLU()
        )
        
        # Multi-scale fusion at bottleneck (PANet approach)
        self.fusion_bottleneck = PANetFusion()
        
        # Fine-scale decoder (preserves crack details)
        self.fine_dec1 = Up(1024, 512)
        self.fine_dec2 = Up(512, 256)
        self.fine_dec3 = Up(256, 128)
        self.fine_dec4 = Up(128, 64)
        self.fine_out = nn.Conv2d(64, num_classes, 1)
        
        # Coarse-scale decoder (compensate for 2x downsampling with extra upsampling)
        self.coarse_dec1 = Up(1024, 512)
        self.coarse_dec2 = Up(512, 256)
        self.coarse_dec3 = Up(256, 128)
        self.coarse_dec4 = Up(128, 64)
        
        # Additional upsampling to match fine stream resolution (compensate for initial 2x downsample)
        self.coarse_final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvNxN(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.coarse_out = nn.Conv2d(64, num_classes, 1)
        
        # Final combination (fine details + coarse validation)
        self.final_conv = nn.Sequential(
            ConvNxN(2, 64, kernel_size=3),
            nn.ReLU(),
            ConvNxN(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        # Initial convolution processing
        fine_inc = self.fine_inc(x)    # [B, 64, 448, 448] - Fine initial features at full resolution
        coarse_inc = self.coarse_inc(x) # [B, 64, 224, 224] - Coarse initial features at 2x downsampled resolution
        
        # Fine-scale encoder path (local crack detection)
        f1, f1_skip = self.fine_enc1(fine_inc)  # [B, 128, 224, 224] - Local patterns
        f2, f2_skip = self.fine_enc2(f1)        # [B, 256, 112, 112] - Crack textures
        f3, f3_skip = self.fine_enc3(f2)        # [B, 512, 56, 56] - Fine features
        f4, f4_skip = self.fine_enc4(f3)        # [B, 1024, 28, 28] - Deep features
        f_bottleneck = self.fine_bottleneck(f4)              # [B, 1024, 28, 28]
        
        # Coarse-scale encoder path with stride=2 downsampling and increasing dilation
        c1, c1_skip = self.coarse_enc1(coarse_inc) # [B, 128, 112, 112] - Stride=2 downsampled, dilation=1
        c2, c2_skip = self.coarse_enc2(c1)         # [B, 256, 56, 56] - Stride=2 downsampled, dilation=2
        c3, c3_skip = self.coarse_enc3(c2)         # [B, 512, 28, 28] - Stride=2 downsampled, dilation=4
        c4, c4_skip = self.coarse_enc4(c3)         # [B, 1024, 14, 14] - Stride=2 downsampled, dilation=8
        c_bottleneck = self.coarse_bottleneck(c4)              # [B, 1024, 14, 14]
        
        # Multi-scale fusion at bottleneck (upsample coarse to match fine resolution)
        c_bottleneck_upsampled = F.interpolate(c_bottleneck, size=f_bottleneck.shape[2:], mode='bilinear', align_corners=True)
        fused_bottleneck = self.fusion_bottleneck(f_bottleneck, c_bottleneck_upsampled)
        
        # Fine-scale decoder (detail preservation)
        fd1 = self.fine_dec1(fused_bottleneck, f4_skip)  # [B, 512, 56, 56]
        fd2 = self.fine_dec2(fd1, f3_skip)               # [B, 256, 112, 112]
        fd3 = self.fine_dec3(fd2, f2_skip)               # [B, 128, 224, 224]
        fd4 = self.fine_dec4(fd3, f1_skip)               # [B, 64, 448, 448]
        
        # Coarse-scale decoder (needs to match skip connection resolutions)
        # Upsample fused bottleneck to match coarse stream's expected input size
        fused_for_coarse = F.interpolate(fused_bottleneck, size=c4_skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply coarse decoder with appropriately sized skip connections
        cd1 = self.coarse_dec1(fused_for_coarse, c4_skip)  # [B, 512, 28, 28]
        
        # Upsample cd1 to match c3_skip size
        cd1_upsampled = F.interpolate(cd1, size=c3_skip.shape[2:], mode='bilinear', align_corners=True)
        cd2 = self.coarse_dec2(cd1_upsampled, c3_skip)     # [B, 256, 56, 56]
        
        # Continue with proper size matching
        cd2_upsampled = F.interpolate(cd2, size=c2_skip.shape[2:], mode='bilinear', align_corners=True)
        cd3 = self.coarse_dec3(cd2_upsampled, c2_skip)     # [B, 128, 112, 112]
        
        cd3_upsampled = F.interpolate(cd3, size=c1_skip.shape[2:], mode='bilinear', align_corners=True)
        cd4 = self.coarse_dec4(cd3_upsampled, c1_skip)     # [B, 64, 224, 224]
        
        # Final upsampling to match fine stream resolution
        cd4_final = self.coarse_final_upsample(cd4)        # [B, 64, 448, 448]
        
        # Generate outputs
        fine_out = self.fine_out(fd4)       # [B, 1, 448, 448] - High sensitivity crack detection
        coarse_out = self.coarse_out(cd4_final)  # [B, 1, 448, 448] - Context-aware validation
        
        # Smart combination: Fine detection + Coarse validation
        fine_prob = torch.sigmoid(fine_out)     # Crack sensitivity
        coarse_prob = torch.sigmoid(coarse_out) # Structural understanding
        
        # Combine: Keep fine detections that coarse stream validates
        combined = torch.cat([fine_prob, coarse_prob], dim=1)
        final_output = self.final_conv(combined)
        
        return final_output