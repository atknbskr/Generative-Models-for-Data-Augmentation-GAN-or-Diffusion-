import os
import sys
sys.path.append(os.getcwd()) # Ensure current directory is in python path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from models.classifier import Classifier
from utils.data_utils import create_imbalanced_dataset

# ==========================================
# LITE GENERATOR (Must match the trained model)
# ==========================================
class LiteGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(LiteGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = 32 // 4
        
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

# ==========================================
# SYNTHETIC DATASET CLASS
# ==========================================
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, generator, minority_classes, samples_per_class, latent_dim, device, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        generator.eval()
        with torch.no_grad():
            for label_idx in minority_classes:
                # Generate in batches
                n_batches = samples_per_class // 100
                for _ in range(n_batches):
                    z = torch.randn(100, latent_dim).to(device)
                    # Correct way to create labels tensor
                    labels = torch.full((100,), label_idx, dtype=torch.long).to(device)
                    
                    imgs = generator(z, labels)
                    imgs = imgs.cpu()
                    
                    # Denormalize to [0,1] then back to PIL-like range if needed,
                    # but here we keep as tensor [-1, 1] usually.
                    # Classifier expects normalized [-1, 1].
                    # Generator output is Tanh -> [-1, 1].
                    # So we can use directly.
                    
                    self.images.append(imgs)
                    self.labels.append(torch.full((100,), label_idx, dtype=torch.long))

        self.images = torch.cat(self.images)
        self.labels = torch.cat(self.labels)
        
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label.item() # Return int to match CIFAR10 dataset

    def __len__(self):
        return len(self.labels)

# ==========================================
# TRAINING FUNCTION
# ==========================================
def train_classifier_lite(config, use_augmentation=False):
    device = torch.device('cpu') # Force CPU
    print(f"Using device: {device}")
    
    save_dir = config['save_dir'] + ('_augmented' if use_augmentation else '_baseline')
    os.makedirs(save_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load Data
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_dataset = create_imbalanced_dataset(
        full_dataset,
        minority_classes=config['minority_classes'],
        minority_ratio=config['minority_ratio']
    )
    
    # Augmentation
    if use_augmentation:
        print("Loading LiteGenerator...")
        generator = LiteGenerator(config['latent_dim'], config['num_classes']).to(device)
        
        # Load weights
        checkpoint = torch.load(config['generator_path'], map_location=device)
        # Check if saved whole model or state dict
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
        else:
            generator.load_state_dict(checkpoint) # fallback
            
        print("Generating synthetic samples...")
        synthetic_dataset = SyntheticDataset(
            generator=generator,
            minority_classes=config['minority_classes'],
            samples_per_class=config['synthetic_samples_per_class'],
            latent_dim=config['latent_dim'],
            device=device
        )
        
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
        print(f"Added {len(synthetic_dataset)} synthetic samples.")

    # Dataloaders - Reduce workers for CPU safety
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Classifier
    classifier = Classifier(num_classes=config['num_classes']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config['lr'])
    
    print(f"Starting Training ({'Augmented' if use_augmentation else 'Baseline'})...")
    
    best_acc = 0.0
    for epoch in range(config['num_epochs']):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        
        # Validation
        classifier.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = classifier(imgs)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()
                
        test_acc = 100. * correct_test / total_test
        print(f"Epoch {epoch+1} - Train Acc: {acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(classifier.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    print(f"Finished. Best Test Acc: {best_acc:.2f}%")
    return best_acc

if __name__ == '__main__':
    config = {
        'num_classes': 10,
        'batch_size': 64,
        'num_epochs': 20, # Reduced to 20 for speed
        'lr': 0.001,
        'minority_classes': [2, 3, 4],
        'minority_ratio': 0.1,
        'latent_dim': 100,
        'synthetic_samples_per_class': 3000,
        'generator_path': './checkpoints/gan/final_model.pth',
        'save_dir': './checkpoints/classifier'
    }
    
    print("="*60)
    print("AI EXPERIMENT STARTED")
    print("Phase 1: Baseline Training (Imbalanced Data)")
    print("="*60)
    acc_baseline = train_classifier_lite(config, use_augmentation=False)
    
    print("\n"+"="*60)
    print("Phase 2: Augmented Training (With GAN Data)")
    print("="*60)
    acc_augmented = train_classifier_lite(config, use_augmentation=True)
    
    print("\n"+"="*60)
    print("FINAL RESULTS")
    print(f"Baseline Accuracy: {acc_baseline:.2f}%")
    print(f"Augmented Accuracy: {acc_augmented:.2f}%")
    print(f"Improvement: {acc_augmented - acc_baseline:.2f}%")
    print("="*60)
