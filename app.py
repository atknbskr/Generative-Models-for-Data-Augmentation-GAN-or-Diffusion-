from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import torch
import numpy as np
import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from models.cgan import Generator, LiteGenerator
from models.classifier import Classifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Global variables for models
generator = None
classifier_baseline = None
classifier_augmented = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_models():
    """Load trained models"""
    global generator, classifier_baseline, classifier_augmented
    
    try:
        # Load generator - try to load best checkpoint (epoch 190) first, then fallback to others
        generator_loaded = False
        
        # Try loading epoch 190 checkpoint (latest and best - most trained)
        epoch_190_path = './checkpoints/gan/lite_generator_190.pth'
        if os.path.exists(epoch_190_path):
            try:
                state_dict = torch.load(epoch_190_path, map_location=device)
                generator = LiteGenerator(latent_dim=100, num_classes=10).to(device)
                generator.load_state_dict(state_dict)
                generator.eval()
                print("Generator loaded from epoch 190 checkpoint (BEST MODEL)")
                generator_loaded = True
            except Exception as e:
                print(f"Failed to load epoch 190 checkpoint: {e}")
        
        # Try epoch 180 as fallback
        if not generator_loaded:
            epoch_180_path = './checkpoints/gan/lite_generator_180.pth'
            if os.path.exists(epoch_180_path):
                try:
                    state_dict = torch.load(epoch_180_path, map_location=device)
                    generator = LiteGenerator(latent_dim=100, num_classes=10).to(device)
                    generator.load_state_dict(state_dict)
                    generator.eval()
                    print("Generator loaded from epoch 180 checkpoint")
                    generator_loaded = True
                except Exception as e:
                    print(f"Failed to load epoch 180 checkpoint: {e}")
        
        # Try epoch 90 as fallback
        if not generator_loaded:
            epoch_90_path = './checkpoints/gan/lite_generator_90.pth'
            if os.path.exists(epoch_90_path):
                try:
                    state_dict = torch.load(epoch_90_path, map_location=device)
                    generator = LiteGenerator(latent_dim=100, num_classes=10).to(device)
                    generator.load_state_dict(state_dict)
                    generator.eval()
                    print("Generator loaded from epoch 90 checkpoint")
                    generator_loaded = True
                except Exception as e:
                    print(f"Failed to load epoch 90 checkpoint: {e}")
        
        # Fallback to final_model.pth
        if not generator_loaded and os.path.exists('./checkpoints/gan/final_model.pth'):
            checkpoint = torch.load('./checkpoints/gan/final_model.pth', map_location=device)
            state_dict = checkpoint['generator'] if isinstance(checkpoint, dict) and 'generator' in checkpoint else checkpoint
            
            try:
                # Try loading as standard Generator
                generator = Generator(latent_dim=100, num_classes=10).to(device)
                generator.load_state_dict(state_dict)
            except Exception as e:
                # If fails, try LiteGenerator
                print(f"Standard Generator load failed ({e}), trying LiteGenerator...")
                generator = LiteGenerator(latent_dim=100, num_classes=10).to(device)
                generator.load_state_dict(state_dict)
            
            generator.eval()
            print("Generator loaded from final_model.pth")
            generator_loaded = True
        
        if not generator_loaded:
            print("Warning: No generator checkpoint found")
        
        # Load baseline classifier
        if os.path.exists('./checkpoints/classifier_baseline/best_model.pth'):
            classifier_baseline = Classifier(num_classes=10).to(device)
            checkpoint = torch.load('./checkpoints/classifier_baseline/best_model.pth', map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            classifier_baseline.load_state_dict(state_dict)
            classifier_baseline.eval()
            print("Baseline classifier loaded successfully")
        
        # Load augmented classifier
        if os.path.exists('./checkpoints/classifier_augmented/best_model.pth'):
            classifier_augmented = Classifier(num_classes=10).to(device)
            checkpoint = torch.load('./checkpoints/classifier_augmented/best_model.pth', map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            classifier_augmented.load_state_dict(state_dict)
            classifier_augmented.eval()
            print("Augmented classifier loaded successfully")
    
    except Exception as e:
        print(f"Error loading models: {e}")


@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate_images():
    """Generate images using the trained GAN"""
    if generator is None:
        return jsonify({'error': 'Generator not loaded. Please train the GAN first.'}), 400
    
    data = request.json
    class_idx = int(data.get('class', 0))
    num_images = int(data.get('num_images', 10))
    
    try:
        with torch.no_grad():
            # Use tighter truncated normal distribution for better quality
            # Clamp noise to [-1.5, 1.5] for more stable and focused generation
            z = torch.clamp(torch.randn(num_images, 100), -1.5, 1.5).to(device)
            labels = torch.full((num_images,), class_idx, dtype=torch.long).to(device)
            
            # Generate multiple samples and select the best ones (optional - for even better quality)
            # For now, just generate normally
            gen_imgs = generator(z, labels)
            
            # Denormalize and convert to images
            gen_imgs = gen_imgs.cpu() * 0.5 + 0.5
            gen_imgs = torch.clamp(gen_imgs, 0, 1)
            
            # Convert to base64 with aggressive post-processing for maximum sharpness
            images_b64 = []
            for img in gen_imgs:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                # Step 1: Upscale to 128x128 using high-quality LANCZOS resampling
                # This makes images appear much sharper and clearer
                pil_img = pil_img.resize((128, 128), Image.Resampling.LANCZOS)
                
                # Step 2: Apply aggressive sharpening (multiple passes for better results)
                # First pass - strong sharpening
                pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                # Second pass - fine detail sharpening
                pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=2))
                
                # Step 3: Enhance contrast more aggressively (1.2 = 20% more contrast)
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1.2)
                
                # Step 4: Enhance brightness slightly to make details more visible
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(1.05)
                
                # Step 5: Enhance color saturation for more vivid colors
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(1.1)
                
                # Step 6: Final sharpening pass
                pil_img = pil_img.filter(ImageFilter.SHARPEN)
                
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                img_b64 = base64.b64encode(buffer.getvalue()).decode()
                images_b64.append(img_b64)
        
        return jsonify({
            'images': images_b64,
            'class_name': CLASS_NAMES[class_idx]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Classify an uploaded image"""
    if classifier_baseline is None and classifier_augmented is None:
        return jsonify({'error': 'Classifiers not loaded. Please train the classifiers first.'}), 400
    
    try:
        # Get image from request
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((32, 32))
        
        # Preprocess
        img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        results = {}
        
        # Classify with baseline
        if classifier_baseline is not None:
            with torch.no_grad():
                output = classifier_baseline(img_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                
                results['baseline'] = {
                    'predicted_class': CLASS_NAMES[pred_class],
                    'confidence': float(probs[pred_class]),
                    'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(10)}
                }
        
        # Classify with augmented
        if classifier_augmented is not None:
            with torch.no_grad():
                output = classifier_augmented(img_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                
                results['augmented'] = {
                    'predicted_class': CLASS_NAMES[pred_class],
                    'confidence': float(probs[pred_class]),
                    'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(10)}
                }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get training results and metrics"""
    results = {
        'baseline': None,
        'augmented': None,
        'comparison': None
    }
    
    try:
        # Load baseline results
        if os.path.exists('./checkpoints/classifier_baseline/results.pth'):
            baseline = torch.load('./checkpoints/classifier_baseline/results.pth', map_location='cpu')
            results['baseline'] = {
                'accuracy': float(baseline['final_accuracy']),
                'f1_score': float(baseline['final_f1']),
                'per_class_accuracy': [float(x) for x in baseline['per_class_accuracy']],
                'class_names': CLASS_NAMES
            }
        
        # Load augmented results
        if os.path.exists('./checkpoints/classifier_augmented/results.pth'):
            augmented = torch.load('./checkpoints/classifier_augmented/results.pth', map_location='cpu')
            results['augmented'] = {
                'accuracy': float(augmented['final_accuracy']),
                'f1_score': float(augmented['final_f1']),
                'per_class_accuracy': [float(x) for x in augmented['per_class_accuracy']],
                'class_names': CLASS_NAMES
            }
        
        # Calculate comparison
        if results['baseline'] and results['augmented']:
            results['comparison'] = {
                'accuracy_improvement': results['augmented']['accuracy'] - results['baseline']['accuracy'],
                'f1_improvement': results['augmented']['f1_score'] - results['baseline']['f1_score'],
                'per_class_improvement': [
                    results['augmented']['per_class_accuracy'][i] - results['baseline']['per_class_accuracy'][i]
                    for i in range(10)
                ]
            }
    
    except Exception as e:
        print(f"Error loading results: {e}")
    
    return jsonify(results)


@app.route('/api/images/<path:filename>')
def get_image(filename):
    """Serve generated images"""
    try:
        return send_file(filename, mimetype='image/png')
    except:
        return jsonify({'error': 'Image not found'}), 404


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model loading status"""
    return jsonify({
        'generator_loaded': generator is not None,
        'baseline_loaded': classifier_baseline is not None,
        'augmented_loaded': classifier_augmented is not None,
        'device': str(device)
    })


if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
