import cv2
import numpy as np
from skimage import io, transform
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import uuid
import time

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create upload folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing functions

def preprocess_image(image_path):
    """
    Preprocess the image to:
    1. Detect edges and dominant lines
    2. Correct orientation based on line detection
    3. Save the processed images
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect dominant lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Find most common angle
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        angles.append(angle)

    # Find most frequent angle
    from collections import Counter
    angle_counts = Counter(np.round(angles, 1))
    dominant_angle = angle_counts.most_common(1)[0][0]

    # Calculate correction angle
    if dominant_angle < 45:
        correction_angle = dominant_angle
    elif dominant_angle < 135:
        correction_angle = dominant_angle - 90
    else:
        correction_angle = dominant_angle - 180

    # Rotate to correct the image orientation
    rotated = transform.rotate(img, -correction_angle)

    # Save the processed image at different stages
    timestamp = int(time.time())
    base_filename = f"{timestamp}_{os.path.basename(image_path)}"
    
    # Save rotated image
    rotated_path = os.path.join(app.config['PROCESSED_FOLDER'], f"rotated_{base_filename}")
    cv2.imwrite(rotated_path, rotated * 255)  # skimage returns float values, multiply by 255 for uint8 format

    return {
        "original": image_path,
        "rotated": rotated_path
    }

# Flask routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Create unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Process the image
            result_paths = preprocess_image(filepath)
            # Pass the paths to the result template
            return render_template('result.html', result=result_paths)
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
