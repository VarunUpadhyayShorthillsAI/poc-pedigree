import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_shadow_and_wrinkles(img):
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_img = clahe.apply(diff_img)
        result_planes.append(norm_img)

    shadow_free = cv2.merge(result_planes)
    wrinkle_reduced = cv2.bilateralFilter(shadow_free, d=9, sigmaColor=75, sigmaSpace=75)
    return wrinkle_reduced

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_document_edges_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            return four_point_transform(img, approx.reshape(4, 2))

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        doc_cnt = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]]
        ])
        return four_point_transform(img, doc_cnt.reshape(4, 2))

    raise ValueError("Document edges not detected")

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
            processed_image_path, info = process_image(filepath)
            return render_template('result.html', result={
                'original': filepath,
                'oriented': processed_image_path,
                'angle_info': info
            })
        except Exception as e:
            return render_template('error.html', error=str(e))
    return redirect(request.url)

def process_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image file could not be read.")

    img = remove_shadow_and_wrinkles(img)
    img = detect_document_edges_and_crop(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    tint_category = (
        "Very Dark" if mean_brightness <= 50 else
        "Dark" if mean_brightness <= 100 else
        "Normal" if mean_brightness <= 180 else
        "Bright"
    )

    processed_image_path = os.path.join(os.path.dirname(image_path), 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, img)

    angle_info = {
        "mean_brightness": round(mean_brightness, 2),
        "tint_category": tint_category
    }

    return processed_image_path, angle_info

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