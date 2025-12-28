"""
Quick Start Script for GAN Data Augmentation Project
This script provides an interactive menu to run different parts of the project
"""

import os
import sys

def print_header():
    print("=" * 60)
    print("  GAN-Based Data Augmentation for Imbalanced Classification")
    print("=" * 60)
    print()

def print_menu():
    print("\nWhat would you like to do?")
    print("1. Train Conditional GAN (Step 1)")
    print("2. Train Classifiers (Step 2 - requires trained GAN)")
    print("3. Launch Web Interface (requires trained models)")
    print("4. Install Dependencies")
    print("5. Check System Requirements")
    print("6. Exit")
    print()

def check_requirements():
    print("\nChecking system requirements...")
    
    # Check Python version
    import sys
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed")
    
    # Check other dependencies
    deps = ['torchvision', 'numpy', 'matplotlib', 'sklearn', 'flask', 'PIL']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep} installed")
        except ImportError:
            print(f"✗ {dep} not installed")
    
    print("\nIf any dependencies are missing, choose option 4 to install them.")

def install_dependencies():
    print("\nInstalling dependencies...")
    os.system("pip install -r requirements.txt")
    print("\nDependencies installed!")

def train_gan():
    print("\n" + "=" * 60)
    print("  Training Conditional GAN")
    print("=" * 60)
    print("\nThis will:")
    print("- Download CIFAR-10 dataset (if not already downloaded)")
    print("- Create an imbalanced version (bird, cat, deer at 10%)")
    print("- Train a conditional GAN for 200 epochs")
    print("- Save checkpoints and sample images")
    print("\nEstimated time: 2-4 hours (GPU) or 10-20 hours (CPU)")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        os.system("python train_gan.py")
    else:
        print("Cancelled.")

def train_classifiers():
    print("\n" + "=" * 60)
    print("  Training Classifiers")
    print("=" * 60)
    print("\nThis will:")
    print("- Train baseline classifier (no augmentation)")
    print("- Train augmented classifier (with synthetic data)")
    print("- Compare performance and generate reports")
    print("\nEstimated time: 1-2 hours (GPU) or 5-10 hours (CPU)")
    print()
    
    # Check if GAN is trained
    if not os.path.exists('./checkpoints/gan/final_model.pth'):
        print("⚠ WARNING: GAN model not found!")
        print("Please train the GAN first (option 1)")
        return
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        os.system("python train_classifier.py")
    else:
        print("Cancelled.")

def launch_web_interface():
    print("\n" + "=" * 60)
    print("  Launching Web Interface")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    print()
    
    os.system("python app.py")

def main():
    while True:
        print_header()
        print_menu()
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            train_gan()
        elif choice == '2':
            train_classifiers()
        elif choice == '3':
            launch_web_interface()
        elif choice == '4':
            install_dependencies()
        elif choice == '5':
            check_requirements()
        elif choice == '6':
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
