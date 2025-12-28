from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import torch
import numpy as np
import io
import base64
from PIL import Image
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
        # Load generator
        if os.path.exists('./checkpoints/gan/final_model.pth'):
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
            print("Generator loaded successfully")
        
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
            z = torch.randn(num_images, 100).to(device)
            labels = torch.full((num_images,), class_idx, dtype=torch.long).to(device)
            gen_imgs = generator(z, labels)
            
            # Denormalize and convert to images
            gen_imgs = gen_imgs.cpu() * 0.5 + 0.5
            gen_imgs = torch.clamp(gen_imgs, 0, 1)
            
            # Convert to base64
            images_b64 = []
            for img in gen_imgs:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
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
