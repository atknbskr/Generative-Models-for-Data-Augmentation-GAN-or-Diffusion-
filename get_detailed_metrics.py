import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
import sys
import json

# Ensure local modules can be imported
sys.path.append(os.getcwd())
try:
    from models.classifier import Classifier
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'models'))
    from classifier import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_metrics(model_path, title):
    print(f"Evaluating {title}...")
    model = Classifier(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
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
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    return report

if __name__ == '__main__':
    baseline_report = get_metrics('./checkpoints/classifier_baseline/best_model.pth', 'Baseline')
    augmented_report = get_metrics('./checkpoints/classifier_augmented/best_model.pth', 'Augmented')
    
    results = {
        'baseline': baseline_report,
        'augmented': augmented_report
    }
    
    print(json.dumps(results, indent=2))
    
    # Save to file to read later if needed
    with open('detailed_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
