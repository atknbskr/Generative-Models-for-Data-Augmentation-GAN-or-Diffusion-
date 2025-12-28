import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.cgan import Generator, Discriminator
from utils.data_utils import create_imbalanced_dataset, save_checkpoint

def train_gan(config):
    """Train conditional GAN on imbalanced dataset - OPTIMIZED FOR SPEED"""
    
    # Device configuration - FORCE CPU if GPU is problematic or just use available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu') # Force CPU to avoid hanging
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    # Data loading - OPTIMIZED: More workers, larger batch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Create imbalanced dataset
    imbalanced_dataset = create_imbalanced_dataset(
        full_dataset, 
        minority_classes=config['minority_classes'],
        minority_ratio=config['minority_ratio']
    )
    
    dataloader = DataLoader(
        imbalanced_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,  # OPTIMIZED: More workers
        pin_memory=True  # OPTIMIZED: Faster data transfer
    )
    
    # Initialize models
    generator = Generator(
        latent_dim=config['latent_dim'],
        num_classes=config['num_classes']
    ).to(device)
    
    discriminator = Discriminator(
        num_classes=config['num_classes']
    ).to(device)
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Optimizers - OPTIMIZED: Higher learning rate
    optimizer_G = optim.Adam(generator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    # Training loop
    print("Starting OPTIMIZED GAN training...")
    print(f"Target: {config['num_epochs']} epochs")
    g_losses = []
    d_losses = []
    
    for epoch in range(config['num_epochs']):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for i, (imgs, labels) in enumerate(pbar):
            batch_size = imgs.size(0)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise and labels
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            gen_labels = torch.randint(0, config['num_classes'], (batch_size,)).to(device)
            
            # Generate images
            gen_imgs = generator(z, gen_labels)
            
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)
            
            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Update progress bar
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
        
        # Average losses
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
        
        # Save sample images
        if (epoch + 1) % config['sample_interval'] == 0:
            save_sample_images(generator, epoch + 1, config, device)
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    save_checkpoint({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, os.path.join(config['save_dir'], 'final_model.pth'))
    
    # Plot training curves
    plot_training_curves(g_losses, d_losses, config['save_dir'])
    
    print("GAN training completed!")
    return generator, discriminator


def save_sample_images(generator, epoch, config, device):
    """Save sample generated images"""
    generator.eval()
    with torch.no_grad():
        # Generate images for each class
        n_row = config['num_classes']
        n_col = 10
        
        z = torch.randn(n_row * n_col, config['latent_dim']).to(device)
        labels = torch.tensor([i for i in range(n_row) for _ in range(n_col)]).to(device)
        
        gen_imgs = generator(z, labels)
        gen_imgs = gen_imgs.cpu()
        
        # Denormalize
        gen_imgs = gen_imgs * 0.5 + 0.5
        gen_imgs = torch.clamp(gen_imgs, 0, 1)
        
        # Create grid
        fig, axes = plt.subplots(n_row, n_col, figsize=(15, 15))
        for i in range(n_row):
            for j in range(n_col):
                idx = i * n_col + j
                img = gen_imgs[idx].permute(1, 2, 0).numpy()
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(f'Class {i}', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['sample_dir'], f'epoch_{epoch}.png'))
        plt.close()
    
    generator.train()


def plot_training_curves(g_losses, d_losses, save_dir):
    """Plot and save training loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


if __name__ == '__main__':
    config = {
        'latent_dim': 100,
        'num_classes': 10,
        'batch_size': 128,  # OPTIMIZED: Larger batch
        'num_epochs': 100,  # OPTIMIZED: Reduced from 200
        'lr': 0.0003,  # OPTIMIZED: Higher learning rate
        'minority_classes': [2, 3, 4],  # bird, cat, deer
        'minority_ratio': 0.1,
        'sample_interval': 20,  # OPTIMIZED: Less frequent
        'checkpoint_interval': 50,
        'save_dir': './checkpoints/gan',
        'sample_dir': './samples/gan'
    }
    
    print("=" * 60)
    print("OPTIMIZED GAN TRAINING - FORCE CPU")
    print("=" * 60)
    print("GPU is hanging due to architecture mismatch (RTX 5060 vs PyTorch/CUDA).")
    print("Switching to OPTIMIZED CPU training to meet deadline.")
    print(f"Epochs: {config['num_epochs']} (reduced for speed)")
    print(f"Batch size: {config['batch_size']} (increased for efficiency)")
    print(f"Learning rate: {config['lr']} (increased for faster convergence)")
    print("=" * 60)
    
    train_gan(config)
