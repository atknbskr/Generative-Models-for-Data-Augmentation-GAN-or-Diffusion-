import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Conditional Generator for CIFAR-10 (32x32 images)"""
    def __init__(self, latent_dim=100, num_classes=10, img_channels=3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        # Initial size after first layer
        self.init_size = 4
        self.fc = nn.Linear(latent_dim + num_classes, 512 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            
            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final layer
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Concatenate noise and label embedding
        label_input = self.label_emb(labels)
        gen_input = torch.cat([noise, label_input], dim=1)
        
        out = self.fc(gen_input)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """Conditional Discriminator for CIFAR-10"""
    def __init__(self, num_classes=10, img_channels=3):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.25))
            return block
        
        self.conv_blocks = nn.Sequential(
            # 32x32 -> 16x16
            *discriminator_block(img_channels + 1, 64, bn=False),
            # 16x16 -> 8x8
            *discriminator_block(64, 128),
            # 8x8 -> 4x4
            *discriminator_block(128, 256),
            # 4x4 -> 2x2
            *discriminator_block(256, 512),
        )
        
        # Calculate output size
        ds_size = 2
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Create label channel to concatenate with image
        # Shape: [batch_size, num_classes] -> [batch_size, 1, H, W]
        label_input = self.label_emb(labels)  # [batch_size, num_classes]
        
        # Create a spatial label map filled with the first embedding value
        # This creates a single channel with constant value per sample
        batch_size = img.size(0)
        label_channel = torch.zeros(batch_size, 1, img.size(2), img.size(3), device=img.device)
        
        # Fill each sample's channel with its label index normalized
        for i in range(batch_size):
            label_channel[i, 0, :, :] = labels[i].float() / self.num_classes
        
        d_in = torch.cat([img, label_channel], dim=1)
        
        out = self.conv_blocks(d_in)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        
        return validity


class LiteGenerator(nn.Module):
    """Lightweight Generator for fast CPU training"""
    def __init__(self, latent_dim=100, num_classes=10):
        super(LiteGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = 32 // 4  # Initial size before upsampling
        
        # Reduced filters: 128 -> 64
        self.l1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 64 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
