import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.data_utils import create_imbalanced_dataset, save_checkpoint
import time

# ==========================================
# LITE MODELS (Smaller for Fast CPU Training)
# ==========================================

class LiteGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes):
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

class LiteDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(LiteDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.num_classes = num_classes

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return nn.Sequential(*block)

        # Reduced filters: 16 -> 32 -> 64 (removed 128 layer)
        self.model = nn.Sequential(
            discriminator_block(3 + 1, 16, bn=False),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 3
        self.adv_layer = nn.Sequential(nn.Linear(64 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, img, labels):
        # Create label channel
        batch_size = img.size(0)
        label_channel = torch.zeros(batch_size, 1, img.size(2), img.size(3), device=img.device)
        for i in range(batch_size):
            label_channel[i, 0, :, :] = labels[i].float() / self.num_classes
            
        d_in = torch.cat([img, label_channel], dim=1)
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# ==========================================
# TRAINING SCRIPT
# ==========================================

def train_gan_lite(config):
    # Force CPU to avoid incompatibility
    device = torch.device('cpu')
    print(f"Using device: {device} (LITE MODE)")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    imbalanced_dataset = create_imbalanced_dataset(
        full_dataset, 
        minority_classes=config['minority_classes'],
        minority_ratio=config['minority_ratio']
    )
    
    dataloader = DataLoader(
        imbalanced_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # Initialize Lite Models
    generator = LiteGenerator(config['latent_dim'], config['num_classes']).to(device)
    discriminator = LiteDiscriminator(config['num_classes']).to(device)
    
    adversarial_loss = nn.BCELoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    print("Starting LITE GAN training...")
    print(f"Target: {config['num_epochs']} epochs")
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            batch_size = imgs.size(0)
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            gen_labels = torch.randint(0, config['num_classes'], (batch_size,)).to(device)
            gen_imgs = generator(z, gen_labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            d_real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
        epoch_duration = time.time() - epoch_start
        print(f"Epoch {epoch+1} done in {epoch_duration:.1f}s. Losses: G={g_loss.item():.4f}, D={d_loss.item():.4f}")
        
        # Save occasionally
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(config['save_dir'], f'lite_generator_{epoch+1}.pth'))
            
    # Save final
    torch.save(generator.state_dict(), os.path.join(config['save_dir'], 'final_model.pth')) # Save as expected name
    torch.save(discriminator.state_dict(), os.path.join(config['save_dir'], 'final_d_model.pth'))
    print("Lite Training Completed!")

if __name__ == '__main__':
    config = {
        'latent_dim': 100,
        'num_classes': 10,
        'batch_size': 64, 
        'num_epochs': 50, # Reduced to 50
        'lr': 0.0002,
        'minority_classes': [2, 3, 4],
        'minority_ratio': 0.1,
        'save_dir': './checkpoints/gan',
        'sample_dir': './samples/gan'
    }
    train_gan_lite(config)
