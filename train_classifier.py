import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.classifier import Classifier
from models.cgan import Generator
from utils.data_utils import create_imbalanced_dataset, save_checkpoint
from utils.synthetic_dataset import SyntheticDataset


def train_classifier(config, use_augmentation=False):
    """Train classifier with or without synthetic data augmentation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = config['save_dir'] + ('_augmented' if use_augmentation else '_baseline')
    os.makedirs(save_dir, exist_ok=True)
    
    # Data loading
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
    
    # Load CIFAR-10
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Create imbalanced dataset
    train_dataset = create_imbalanced_dataset(
        full_dataset,
        minority_classes=config['minority_classes'],
        minority_ratio=config['minority_ratio']
    )
    
    # Add synthetic data if augmentation is enabled
    if use_augmentation:
        print("Loading generator and creating synthetic samples...")
        generator = Generator(
            latent_dim=config['latent_dim'],
            num_classes=config['num_classes']
        ).to(device)
        
        # Load trained generator
        checkpoint = torch.load(config['generator_path'], map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        generator.eval()
        
        # Generate synthetic samples for minority classes
        synthetic_dataset = SyntheticDataset(
            generator=generator,
            minority_classes=config['minority_classes'],
            samples_per_class=config['synthetic_samples_per_class'],
            latent_dim=config['latent_dim'],
            device=device,
            transform=transform
        )
        
        # Combine real and synthetic data
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
        print(f"Added {len(synthetic_dataset)} synthetic samples")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Initialize classifier
    classifier = Classifier(num_classes=config['num_classes']).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training loop
    print(f"Starting classifier training ({'with' if use_augmentation else 'without'} augmentation)...")
    train_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for imgs, labels in pbar:
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
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        test_acc, test_f1, per_class_acc = evaluate_classifier(classifier, test_loader, device, config['num_classes'])
        test_accs.append(test_acc)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'accuracy': test_acc,
                'f1_score': test_f1
            }, os.path.join(save_dir, 'best_model.pth'))
    
    # Final evaluation
    print("\nFinal Evaluation:")
    classifier.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'))['model'])
    final_acc, final_f1, per_class_acc = evaluate_classifier(
        classifier, test_loader, device, config['num_classes'], detailed=True
    )
    
    # Save results
    results = {
        'final_accuracy': final_acc,
        'final_f1': final_f1,
        'per_class_accuracy': per_class_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'use_augmentation': use_augmentation
    }
    
    torch.save(results, os.path.join(save_dir, 'results.pth'))
    
    # Plot training curves
    plot_classifier_curves(train_losses, train_accs, test_accs, save_dir)
    
    # Generate confusion matrix
    generate_confusion_matrix(classifier, test_loader, device, config['num_classes'], save_dir)
    
    return results


def evaluate_classifier(classifier, test_loader, device, num_classes, detailed=False):
    """Evaluate classifier and return metrics"""
    classifier.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = classifier(imgs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    if detailed:
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print(f"Weighted F1-Score: {f1:.4f}")
        print("\nPer-class Accuracy:")
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
            print(f"  Class {i} ({name}): {acc:.2f}%")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return accuracy, f1, per_class_acc


def plot_classifier_curves(train_losses, train_accs, test_accs, save_dir):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


def generate_confusion_matrix(classifier, test_loader, device, num_classes, save_dir):
    """Generate and save confusion matrix"""
    classifier.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = classifier(imgs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    config = {
        'num_classes': 10,
        'batch_size': 128,
        'num_epochs': 100,
        'lr': 0.001,
        'minority_classes': [2, 3, 4],  # bird, cat, deer
        'minority_ratio': 0.1,
        'latent_dim': 100,
        'synthetic_samples_per_class': 4000,
        'generator_path': './checkpoints/gan/final_model.pth',
        'save_dir': './checkpoints/classifier'
    }
    
    # Train baseline classifier
    print("=" * 50)
    print("Training BASELINE classifier (no augmentation)")
    print("=" * 50)
    baseline_results = train_classifier(config, use_augmentation=False)
    
    # Train augmented classifier
    print("\n" + "=" * 50)
    print("Training AUGMENTED classifier (with synthetic data)")
    print("=" * 50)
    augmented_results = train_classifier(config, use_augmentation=True)
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"Baseline Accuracy: {baseline_results['final_accuracy']:.2f}%")
    print(f"Augmented Accuracy: {augmented_results['final_accuracy']:.2f}%")
    print(f"Improvement: {augmented_results['final_accuracy'] - baseline_results['final_accuracy']:.2f}%")
    print(f"\nBaseline F1-Score: {baseline_results['final_f1']:.4f}")
    print(f"Augmented F1-Score: {augmented_results['final_f1']:.4f}")
