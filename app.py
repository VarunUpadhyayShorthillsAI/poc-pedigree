import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, send_from_directory
from PIL import Image
from engine import remove_bg_mult

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
            final_path, info = process_image(filepath)
            return render_template('result.html', result={
                'original': info["original"],
                'intermediate': info["intermediate"],
                'oriented': info["final"]
            })
        except Exception as e:
            return render_template('error.html', error=str(e))

    return redirect(request.url)


def remove_shadow_and_wrinkles(img):
    """Removes shadows and suppresses wrinkles using morphological ops and bilateral filtering."""
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        background = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, background)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result_planes.append(norm)

    shadow_free = cv2.merge(result_planes)

    # Optional wrinkle suppression: bilateral filter
    wrinkle_free = cv2.bilateralFilter(shadow_free, d=9, sigmaColor=75, sigmaSpace=75)
    
    return wrinkle_free


def process_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        # Step 1: Remove shadows and wrinkles using OpenCV
        img_cv = cv2.imread(image_path)
        cleaned = remove_shadow_and_wrinkles(img_cv)

        # Step 2: Save intermediate cleaned image
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        cleaned_pil = Image.fromarray(cleaned_rgb).convert("RGBA")

        intermediate_filename = 'intermediate_' + os.path.splitext(os.path.basename(image_path))[0] + '.png'
        intermediate_path = os.path.join(os.path.dirname(image_path), intermediate_filename)
        cleaned_pil.save(intermediate_path, format='PNG')

        # Step 3: Remove background using U²-Net
        output_image = remove_bg_mult(cleaned_pil)

        # Step 4: Save final output image
        final_filename = 'bg_removed_' + os.path.splitext(os.path.basename(image_path))[0] + '.png'
        final_path = os.path.join(os.path.dirname(image_path), final_filename)
        output_image.save(final_path, format='PNG')

        return final_path, {
            "original": image_path,
            "intermediate": intermediate_path,
            "final": final_path,
            "message": "Processing complete."
        }

    except Exception as err:
        raise RuntimeError(f"Image processing failed: {err}")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/error')
def error():
    return render_template('error.html', error="An error occurred during processing.")


if __name__ == '__main__':
    app.run(debug=True)
