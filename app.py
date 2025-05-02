import os
import cv2
import numpy as np
from skimage import transform
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

import pytesseract
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            rotated_image_path = process_image(filepath)
            return render_template('result.html', result={
                'original': filepath,
                'oriented': rotated_image_path
            })
        except Exception as e:
            return render_template('error.html', error=str(e))

    return redirect(request.url)



def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("The image file was not found!")

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Step 3: Hough line transform to find predominant angles
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        angles = []
        for rho, theta in lines[:,0]:
            angle = (theta * 180 / np.pi) - 90  # shift to [-90, 90]
            if -45 < angle < 45:
                angles.append(angle)
        if angles:
            median_angle = np.median(angles)
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Step 4: OCR orientation detection using pytesseract
    try:
        osd = pytesseract.image_to_osd(Image.fromarray(img), output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle != 0:
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except:
        pass  # Fallback silently if OCR fails

    # Save final rotated image
    rotated_image_path = os.path.join(os.path.dirname(image_path), 'rotated_' + os.path.basename(image_path))
    cv2.imwrite(rotated_image_path, img)
    return rotated_image_path


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/error')
def error():
    return render_template('error.html', error="An error occurred during processing.")

if __name__ == '__main__':
    app.run(debug=True)