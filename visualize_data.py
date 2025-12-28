import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for saving images
save_dir = 'original_samples'
os.makedirs(save_dir, exist_ok=True)

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to show and save image
def save_image_grid():
    # Get some random training images
    # We will get 10 images, one for each class for demonstration
    
    images_per_class = {}
    
    print("Dataset taranıyor...")
    
    # Find one image for each class
    for img, label in trainset:
        if label not in images_per_class:
            images_per_class[label] = img
        if len(images_per_class) == 10:
            break
            
    # Create the plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    print(f"Örnekler {save_dir} klasörüne kaydediliyor...")
    
    for i in range(10):
        img = images_per_class[i]
        # Unnormalize (if it was normalized, but here we just loaded as tensor)
        npimg = img.numpy()
        # Transpose from (C, H, W) to (H, W, C)
        plt_img = np.transpose(npimg, (1, 2, 0))
        
        # Save individual image
        class_name = classes[i]
        plt.imsave(os.path.join(save_dir, f'{i}_{class_name}.png'), plt_img)
        
        # Add to grid
        axes[i].imshow(plt_img)
        axes[i].set_title(class_name)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_classes_grid.png'))
    print("Tamamlandı! 'original_samples' klasörüne bakabilirsiniz.")

if __name__ == '__main__':
    save_image_grid()
