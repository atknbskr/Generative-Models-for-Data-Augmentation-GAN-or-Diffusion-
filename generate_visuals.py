import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Ensure paths
sys.path.append(os.getcwd())
from models.cgan import LiteGenerator
from models.classifier import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = './report_assets'
os.makedirs(save_dir, exist_ok=True)

def generate_gan_samples():
    print("Generating GAN samples...")
    generator = LiteGenerator(num_classes=10).to(device)
    
    # Load model
    path = './checkpoints/gan/final_model.pth'
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['generator'] if isinstance(checkpoint, dict) and 'generator' in checkpoint else checkpoint
        generator.load_state_dict(state_dict)
        generator.eval()
        
        # Generate 10 images for minority classes (Bird, Cat, Deer)
        # Class indices: 2, 3, 4
        z = torch.randn(15, 100).to(device)
        labels = torch.tensor([2]*5 + [3]*5 + [4]*5).to(device)
        
        with torch.no_grad():
            imgs = generator(z, labels).cpu()
            
        # Plot
        fig, axes = plt.subplots(3, 5, figsize=(10, 6))
        class_names = {2: 'Bird', 3: 'Cat', 4: 'Deer'}
        
        for i, ax in enumerate(axes.flat):
            img = imgs[i].permute(1, 2, 0).numpy()
            img = (img * 0.5 + 0.5) # Denormalize
            ax.imshow(np.clip(img, 0, 1))
            ax.axis('off')
            if i % 5 == 2:
                ax.set_title(class_names[labels[i].item()])
                
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gan_samples.png', dpi=300)
        plt.close()

def generate_confusion_matrix():
    print("Generating Confusion Matrix...")
    classifier = Classifier(num_classes=10).to(device)
    path = './checkpoints/classifier_augmented/best_model.pth'
    
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        classifier.load_state_dict(state_dict)
        classifier.eval()
        
        # Load test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
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
        
        # Plot
        plt.figure(figsize=(10, 8))
        class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Augmented Model)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix_augmented.png', dpi=300)
        plt.close()

def generate_confusion_matrix_baseline():
    print("Generating Baseline Confusion Matrix...")
    classifier = Classifier(num_classes=10).to(device)
    path = './checkpoints/classifier_baseline/best_model.pth'
    
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        classifier.load_state_dict(state_dict)
        classifier.eval()
        
        # Load test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
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
        
        # Plot
        plt.figure(figsize=(10, 8))
        class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Baseline Model)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix_baseline.png', dpi=300)
        plt.close()

def generate_comparison_chart():
    print("Generating comparison chart...")
    # Using hardcoded values from our results to ensure consistency
    metrics = ['Accuracy', 'F1-Score']
    baseline = [68.32, 67.0]
    augmented = [69.89, 69.0]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, baseline, width, label='Baseline', color='#6366f1')
    rects2 = ax.bar(x + width/2, augmented, width, label='Augmented', color='#8b5cf6')
    
    ax.set_ylabel('Scores (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(60, 75)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_chart.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    generate_gan_samples()
    generate_confusion_matrix() # Renamed to augmented in previous step implicitly by filename change, keeping name consistent
    generate_confusion_matrix_baseline()
    generate_comparison_chart()
    print("All visuals generated!")
