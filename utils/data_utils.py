import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import os


def create_imbalanced_dataset(dataset, minority_classes, minority_ratio=0.1):
    """
    Create an imbalanced version of the dataset
    
    Args:
        dataset: Original dataset
        minority_classes: List of class indices to make minority
        minority_ratio: Ratio of samples to keep for minority classes (0-1)
    
    Returns:
        Subset of the dataset with imbalanced classes
    """
    targets = np.array(dataset.targets)
    indices = []
    
    for class_idx in range(len(np.unique(targets))):
        class_indices = np.where(targets == class_idx)[0]
        
        if class_idx in minority_classes:
            # Keep only minority_ratio of samples
            n_samples = int(len(class_indices) * minority_ratio)
            selected_indices = np.random.choice(class_indices, n_samples, replace=False)
        else:
            # Keep all samples for majority classes
            selected_indices = class_indices
        
        indices.extend(selected_indices.tolist())
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Print class distribution
    print("\nClass distribution in imbalanced dataset:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    for class_idx in range(len(np.unique(targets))):
        count = sum(targets[indices] == class_idx)
        status = "(MINORITY)" if class_idx in minority_classes else "(MAJORITY)"
        print(f"  Class {class_idx} ({class_names[class_idx]}): {count} samples {status}")
    
    return Subset(dataset, indices)


def save_checkpoint(state, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def get_class_distribution(dataset):
    """Get class distribution of a dataset"""
    if isinstance(dataset, Subset):
        targets = np.array([dataset.dataset.targets[i] for i in dataset.indices])
    else:
        targets = np.array(dataset.targets)
    
    unique, counts = np.unique(targets, return_counts=True)
    return dict(zip(unique, counts))


def calculate_class_weights(dataset, num_classes):
    """Calculate class weights for weighted loss"""
    distribution = get_class_distribution(dataset)
    
    total_samples = sum(distribution.values())
    weights = []
    
    for i in range(num_classes):
        if i in distribution:
            weight = total_samples / (num_classes * distribution[i])
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.FloatTensor(weights)
