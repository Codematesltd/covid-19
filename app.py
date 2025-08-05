from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash
from supabase import create_client
from dotenv import load_dotenv
import os
import hashlib
from functools import wraps
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import random
import io
from llm_integration import llm_bp
from llm_integration.utils import image_to_base64, prepare_image_for_llm
import requests
import functools
import time
import requests.exceptions
from typing import Callable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def call_llm_service(url, payload, headers, timeout=10, max_retries=3, backoff_factor=0.5):
    """Call LLM service with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    try:
        response = session.post(url, json=payload, headers=headers, timeout=timeout)
        return response
    except requests.exceptions.RequestException as e:
        print(f"LLM service call failed: {e}")
        raise

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-123')  # Add secret key for sessions

# Load environment variables from .env
load_dotenv()

# Supabase setup with validation
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Environment variables SUPABASE_URL and SUPABASE_KEY must be set")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Constants
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms for prediction (only basic preprocessing)
transforms_predict = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Model setup
def load_model():
    try:
        checkpoint = torch.load('best_covid_model.pth', map_location=DEVICE)
        
        # Initialize model with pretrained weights
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 2)  # Binary classification
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Model loaded successfully!")
        return model.to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
def tensor_to_img(tensor):
    """Convert tensor to PIL Image"""
    img = tensor.cpu().squeeze(0)  # remove batch dimension
    img = img.permute(1, 2, 0)  # CHW to HWW
    img = img * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN)  # denormalize
    img = (img * 255).clamp(0, 255).numpy().astype(np.uint8)
    return Image.fromarray(img)

def preprocess_image(image_bytes):
    """Standard EfficientNet preprocessing for model input"""
    try:
        # 4. Convert PIL image to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # 1. Resize image to 224x224, 2. Convert to tensor, 3. Normalize
        transforms_predict = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        tensor = transforms_predict(image)
        # 5. Add batch dimension
        tensor = tensor.unsqueeze(0)
        # 6. Move tensor to DEVICE
        tensor = tensor.to(DEVICE)
        return tensor, image
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        raise

# Load model at startup
model = load_model()

# Register LLM blueprint
app.register_blueprint(llm_bp, url_prefix='/llm')

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('sign_in'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('predict.html', error="No file uploaded")
            
            file = request.files['file']
            if file.filename == '':
                return render_template('predict.html', error="No file selected")
            
            # Read image and convert to tensor
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Direct tensor conversion for model
            tensor = transforms_predict(image).unsqueeze(0)
            
            # Save original image for display
            display_path = os.path.join('static', 'preprocessed.jpg')
            image.save(display_path)
            
            # Get prediction with model
            with torch.no_grad():
                outputs = model(tensor.to(DEVICE))
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                pred_probs, predictions = torch.max(probabilities, 1)
                
                # Get probabilities
                covid_prob = float(probabilities[0][0]) * 100
                normal_prob = float(probabilities[0][1]) * 100
                confidence_value = float(pred_probs[0]) * 100
                prediction = int(predictions[0])
                
                print(f"Debug - COVID: {covid_prob}%, Normal: {normal_prob}%, Conf: {confidence_value}%")
                
                # Send raw image to LLM
                image_base64 = image_to_base64(image)
                
                # LLM analysis with retry
                llm_analysis = None
                try:
                    llm_payload = {
                        "prediction": 'COVID' if prediction == 0 else 'Normal',
                        "confidence": confidence_value,
                        "covid_prob": covid_prob,
                        "normal_prob": normal_prob,
                        "image_data": image_base64,
                        "symptoms": []
                    }
                    
                    llm_response = call_llm_service(
                        f"{request.url_root}llm/analyze",
                        llm_payload,
                        {'Content-Type': 'application/json'}
                    )
                    
                    if llm_response.status_code == 200:
                        llm_data = llm_response.json()
                        llm_analysis = llm_data.get('llm_analysis', {})
                    else:
                        print(f"LLM request failed with status: {llm_response.status_code}")
                        
                except requests.exceptions.Timeout:
                    print("LLM request timed out")
                except requests.exceptions.RequestException as e:
                    print(f"LLM request failed: {str(e)}")
                
                result = {
                    'prediction': 'COVID' if prediction == 0 else 'Normal',
                    'confidence': f"{confidence_value:.2f}%",
                    'covid_prob': f"{covid_prob:.2f}%",
                    'normal_prob': f"{normal_prob:.2f}%",
                    'preprocessed_image': 'preprocessed.jpg',
                    'llm_analysis': llm_analysis
                }
                
                return render_template('predict.html', result=result, user=session.get('user'))
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return render_template('predict.html', error=f"Error processing image: {str(e)}", user=session.get('user'))
            
    return render_template('predict.html', user=session.get('user'))

@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        try:
            # Get form data
            full_name = request.form.get('full_name')
            email = request.form.get('email')
            institution = request.form.get('institution')
            password = request.form.get('password')
            confirm = request.form.get('confirm_password')

            # Validate required fields
            if not all([full_name, email, password, confirm]):
                return render_template('sign-up.html', error="All fields are required")

            if password != confirm:
                return render_template('sign-up.html', error="Passwords do not match")

            # Hash password
            password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()

            # Create user
            user_data = {
                "full_name": full_name,
                "email": email,
                "institution": institution,
                "password_hash": password_hash
            }
            
            resp = supabase.table("users").insert(user_data).execute()
            
            if not resp.data:
                return render_template('sign-up.html', error="Failed to create account")
                
            return redirect(url_for('sign_in'))
            
        except Exception as e:
            if "duplicate key value violates unique constraint" in str(e):
                return render_template('sign-up.html', error="Email already registered")
            return render_template('sign-up.html', error="Sign up failed")
            
    return render_template('sign-up.html')

@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        print(f"Attempt login: {email}")  # debug

        if not email or not password:
            return render_template('sign-in.html', error="Email and password are required")

        password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
        try:
            resp = supabase.table("users").select("*").eq("email", email).single().execute()
            print(f"Supabase response: {resp}")  # debug
        except Exception as e:
            print(f"Supabase query error: {e}")
            return render_template('sign-in.html', error="Unable to verify credentials now")

        user_data = None
        # supabase-py v2 returns resp.data directly; adjust if resp.data is a list or dict
        if hasattr(resp, 'data'):
            # if single().execute() returns a list wrapper
            if isinstance(resp.data, list):
                user_data = resp.data[0] if resp.data else None
            else:
                user_data = resp.data
        else:
            # fallback if resp is tuple or list
            user_data = resp[0] if isinstance(resp, (list, tuple)) and resp else None

        if not user_data:
            print("User not found")  # debug
            return render_template('sign-in.html', error="Invalid email or password")

        stored_hash = user_data.get('password_hash')
        if stored_hash != password_hash:
            print("Password mismatch")  # debug
            return render_template('sign-in.html', error="Invalid email or password")

        session['user'] = {
            'email': email,
            'name': user_data.get('full_name')
        }
        print("Login success")  # debug
        return redirect(url_for('predict'))

    return render_template('sign-in.html')
@app.route('/about')
def about():
    return render_template('about.html')   
@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('sign_in'))

if __name__ == '__main__':
    app.run(debug=True)