import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import os
import sys

# Ensure local modules can be imported
sys.path.append(os.getcwd())
try:
    from models.classifier import Classifier
except ImportError:
    # Fallback if running from a different directory
    sys.path.append(os.path.join(os.getcwd(), 'models'))
    from classifier import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_and_save(model_path, save_dir):
    print(f"Processing {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load model
    model = Classifier(num_classes=10).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle state dict format
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate metrics
    import numpy as np
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    per_class_acc = []
    for i in range(10):
        mask = (all_labels == i)
        if mask.sum() > 0:
            class_acc = accuracy_score(all_labels[mask], all_preds[mask]) * 100
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
            
    results = {
        'final_accuracy': acc,
        'final_f1': f1,
        'per_class_accuracy': per_class_acc
    }
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(results, os.path.join(save_dir, 'results.pth'))
    print(f"Saved results to {save_dir}/results.pth (Acc: {acc:.2f}%)")

if __name__ == '__main__':
    # Fix Baseline
    evaluate_and_save(
        './checkpoints/classifier_baseline/best_model.pth',
        './checkpoints/classifier_baseline'
    )
    
    # Fix Augmented
    evaluate_and_save(
        './checkpoints/classifier_augmented/best_model.pth',
        './checkpoints/classifier_augmented'
    )
