import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from PIL import Image
import pytesseract

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
            rotated_image_path, angle_info = process_image(filepath)
            return render_template('result.html', result={
                'original': filepath,
                'oriented': rotated_image_path,
                'angle_info': angle_info
            })
        except Exception as e:
            return render_template('error.html', error=str(e))

    return redirect(request.url)

# def estimate_skew_by_contours(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9, 9), 0)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     coords = np.column_stack(np.where(thresh > 0))
#     if coords.shape[0] < 50:
#         return 0

#     mean, eigenvectors = cv2.PCACompute(coords.astype(np.float32), mean=None)
#     angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
#     return angle

def process_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
 
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image file could not be read. Possibly corrupt or unsupported format.")
 
        # Step 1: Convert to grayscale
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise RuntimeError(f"Error converting to grayscale: {e}")
 
        # Step 2: Edge detection
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        except Exception as e:
            raise RuntimeError(f"Error in edge detection: {e}")
 
        # Step 3: Improved Hough transform for skew detection
        detected_angle = None
        corrected_angle = None
        try:
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is not None and len(lines) > 0:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle_degrees = np.degrees(theta) - 90
                    
                    if angle_degrees < -90:
                        angle_degrees += 180
                    elif angle_degrees > 90:
                        angle_degrees -= 180
                        
                    angles.append(angle_degrees)
                
                if angles:
                    hist, bins = np.histogram(angles, bins=180, range=(-90, 90))
                    most_common_angle_idx = np.argmax(hist)
                    detected_angle = bins[most_common_angle_idx] + (bins[1] - bins[0])/2
                    
                    print(f"Detected document rotation angle: {detected_angle:.2f} degrees")
                    
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
 
                    # Correction logic: Keep angles between -45 and 45
                    corrected_angle = detected_angle
                    if detected_angle < -45:
                        corrected_angle += 180
                    elif detected_angle > 45:
                        corrected_angle -= 180
 
                    print(f"Corrected angle after normalization: {corrected_angle:.2f} degrees")
                    
                    M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            print(f"Error in Hough transform: {e}")
            # Continue processing even if Hough transform fails
 
        # Step 4: OCR orientation detection as backup
        ocr_angle = None
        try:
            osd = pytesseract.image_to_osd(Image.fromarray(img), output_type=pytesseract.Output.DICT)
            ocr_angle = osd.get("rotate", 0)
            
            if ocr_angle not in [0, None]:
                print(f"OCR detected orientation adjustment: {ocr_angle} degrees")
                
                if abs(ocr_angle) >= 20:
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, -ocr_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract not found. Make sure it's installed and in your system PATH.")
        except Exception as e:
            print(f"OCR orientation detection failed (continuing without it): {e}")
 
        # Step 5: Save result and return information
        try:
            angle_info = {
                "hough_angle": round(detected_angle, 2) if detected_angle is not None else None,
                "corrected_angle": round(corrected_angle, 2) if corrected_angle is not None else None,
                "ocr_angle": ocr_angle
            }
            
            rotated_image_path = os.path.join(os.path.dirname(image_path), 'rotated_' + os.path.basename(image_path))
            cv2.imwrite(rotated_image_path, img)
            
            return rotated_image_path, angle_info
        except Exception as e:
            raise RuntimeError(f"Error saving rotated image: {e}")
 
    except Exception as err:
        raise RuntimeError(f"Image processing failed: {err}")
 

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/error')
def error():
    return render_template('error.html', error="An error occurred during processing.")

@app.route('/check_angle', methods=['POST'])
def check_angle():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            _, angle_info = process_image(filepath)
            return jsonify({
                'success': True,
                'angle_info': angle_info
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
