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

# Import Lite models from the existing file
from train_gan_cpu_lite import LiteGenerator, LiteDiscriminator

def train_gan_continue(config, resume_from_epoch=50, additional_epochs=30):
    """Continue training GAN from existing checkpoint"""
    device = torch.device('cpu')
    print(f"Using device: {device} (LITE MODE - CONTINUING TRAINING)")
    
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
    
    # Try to load existing checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(config['save_dir'], f'lite_generator_{resume_from_epoch}.pth')
    discriminator_path = os.path.join(config['save_dir'], 'final_d_model.pth')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading generator from epoch {resume_from_epoch}...")
        generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
        start_epoch = resume_from_epoch
        print(f"[OK] Generator loaded from epoch {start_epoch}")
    elif os.path.exists(os.path.join(config['save_dir'], 'final_model.pth')):
        print("Loading from final_model.pth...")
        checkpoint = torch.load(os.path.join(config['save_dir'], 'final_model.pth'), map_location=device)
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
        else:
            generator.load_state_dict(checkpoint)
        start_epoch = resume_from_epoch
        print(f"[OK] Generator loaded from final_model.pth")
    else:
        print("[WARNING] No checkpoint found, starting from scratch")
    
    if os.path.exists(discriminator_path):
        print("Loading discriminator...")
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
        print("[OK] Discriminator loaded")
    
    adversarial_loss = nn.BCELoss()
    
    # Use learning rate scheduling for better convergence
    # Start with lower LR for fine-tuning, will decay over time
    initial_lr = config['lr'] * 0.3  # Lower initial LR for stability
    optimizer_G = optim.Adam(generator.parameters(), lr=initial_lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=initial_lr, betas=(0.5, 0.999))
    
    # Learning rate scheduler - decay every 20 epochs
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.9)
    
    total_epochs = start_epoch + additional_epochs
    print(f"\n{'='*60}")
    print(f"Continuing LITE GAN training")
    print(f"Starting from epoch: {start_epoch}")
    print(f"Training for: {additional_epochs} more epochs")
    print(f"Total epochs will be: {total_epochs}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")):
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
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
        
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        epoch_duration = time.time() - epoch_start
        
        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()
        current_lr = optimizer_G.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{total_epochs} - {epoch_duration:.1f}s - "
              f"G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), 
                      os.path.join(config['save_dir'], f'lite_generator_{epoch+1}.pth'))
            print(f"  [SAVED] Checkpoint saved at epoch {epoch+1}")
    
    # Save final
    final_checkpoint = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }
    torch.save(final_checkpoint, os.path.join(config['save_dir'], 'final_model.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config['save_dir'], 'final_d_model.pth'))
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[OK] Training Completed!")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Final model saved. Total epochs: {total_epochs}")
    print(f"{'='*60}")

if __name__ == '__main__':
    config = {
        'latent_dim': 100,
        'num_classes': 10,
        'batch_size': 64, 
        'lr': 0.0002,
        'minority_classes': [2, 3, 4],
        'minority_ratio': 0.1,
        'save_dir': './checkpoints/gan',
        'sample_dir': './samples/gan'
    }
    
    # Continue from epoch 90, train 100 more epochs (total: 190)
    # This will significantly improve image quality
    train_gan_continue(config, resume_from_epoch=90, additional_epochs=100)

