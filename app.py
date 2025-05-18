import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from flask import Flask, render_template, request, redirect, send_from_directory
from engine import remove_bg_mult  # U²-Net based background remover

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# File upload and processing endpoint
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
        print(f"📥 Uploaded file saved to: {filepath}")

        try:
            final_path, info = process_image(filepath)
            return render_template('result.html', result={
                'original': info["original"],
                'intermediate': info["intermediate"],
                'oriented': info["final"],
                'angles': info["angles"]
            })
        except Exception as e:
            return render_template('error.html', error=str(e))

    return redirect(request.url)

# Shadow and wrinkle removal using morphological and bilateral filtering
def remove_shadow_and_wrinkles(img):
    print("🔧 Removing shadows and wrinkles...")
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        background = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, background)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result_planes.append(norm)

    shadow_free = cv2.merge(result_planes)
    wrinkle_free = cv2.bilateralFilter(shadow_free, d=9, sigmaColor=75, sigmaSpace=75)
    return wrinkle_free

# Detect rotation using Hough Transform and apply correction
def detect_and_correct_rotation(image: np.ndarray):
    print("🔁 Starting angle detection and rotation correction...")
    try:
        original = image.copy()
        detected_angle = None
        corrected_angle = None

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        angles = []

        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle_deg = np.degrees(theta) - 90
                if angle_deg < -90:
                    angle_deg += 180
                elif angle_deg > 90:
                    angle_deg -= 180
                if -40 <= angle_deg <= 40:
                    angles.append(angle_deg)

            if angles:
                # Find most frequent angle
                hist, bins = np.histogram(angles, bins=180, range=(-90, 90))
                idx = np.argmax(hist)
                detected_angle = bins[idx] + (bins[1] - bins[0]) / 2
                corrected_angle = detected_angle

                print(f"📏 Hough detected angle: {detected_angle:.2f}°")

                # Normalize extreme angles
                if corrected_angle < -90:
                    corrected_angle += 180
                elif corrected_angle > 90:
                    corrected_angle -= 180

                print(f"🔧 Normalized corrected angle: {corrected_angle:.2f}°")

                # Rotate the image
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        else:
            print("⚠️ No lines detected using Hough Transform.")

        return image, {
            "hough_angle": round(detected_angle, 2) if detected_angle is not None else None,
            "corrected_angle": round(corrected_angle, 2) if corrected_angle is not None else None
        }

    except Exception as e:
        print(f"❌ Rotation detection failed: {e}")
        return image, {
            "hough_angle": None,
            "corrected_angle": None
        }

# Use OCR to determine if the image is upside down by analyzing text density
def detect_inversion_by_text_density(image):
    print("🔍 Checking for inversion using text density...")
    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        top_text = 0
        bottom_text = 0

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if len(text) >= 3:
                y = data['top'][i]
                if y < h / 2:
                    top_text += 1
                else:
                    bottom_text += 1

        print(f"🧠 Text lines — Top: {top_text}, Bottom: {bottom_text}")
        return bottom_text > top_text  # If bottom text is denser, likely inverted

    except Exception as e:
        print(f"⚠️ Density check failed: {e}")
        return False

# Complete pipeline for image preprocessing
def process_image(image_path):
    try:
        print(f"🖼 Processing image: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        img_cv = cv2.imread(image_path)

        # STEP 1: Initial rotation (pre-cleaning)
        early_rotated, early_info = detect_and_correct_rotation(img_cv)

        # STEP 2: Shadow and wrinkle removal
        cleaned = remove_shadow_and_wrinkles(early_rotated)

        # STEP 3: Save cleaned intermediate image
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        cleaned_pil = Image.fromarray(cleaned_rgb).convert("RGBA")
        intermediate_filename = 'intermediate_' + os.path.splitext(os.path.basename(image_path))[0] + '.png'
        intermediate_path = os.path.join(os.path.dirname(image_path), intermediate_filename)
        cleaned_pil.save(intermediate_path, format='PNG')
        print(f"📤 Intermediate image saved to: {intermediate_path}")

        # STEP 4: Background removal using U²-Net
        print("🪄 Removing background using U²-Net...")
        bg_removed_pil = remove_bg_mult(cleaned_pil)
        bg_removed_np = np.array(bg_removed_pil.convert("RGB"))

        # STEP 5: Final rotation after background removal
        rotated_final_np, final_info = detect_and_correct_rotation(bg_removed_np)

        # STEP 6: Inversion correction using text OCR
        if detect_inversion_by_text_density(rotated_final_np):
            print("🔄 Inversion detected — applying 180° correction...")
            h, w = rotated_final_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            rotated_final_np = cv2.warpAffine(rotated_final_np, M, (w, h),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE)

        # STEP 7: Save final processed image
        rotated_pil = Image.fromarray(rotated_final_np).convert("RGBA")
        final_filename = 'bg_removed_' + os.path.splitext(os.path.basename(image_path))[0] + '.png'
        final_path = os.path.join(os.path.dirname(image_path), final_filename)
        rotated_pil.save(final_path, format='PNG')
        print(f"✅ Final image saved to: {final_path}")

        return final_path, {
            "original": image_path,
            "intermediate": intermediate_path,
            "final": final_path,
            "angles": {
                "early_rotation": early_info,
                "final_rotation": final_info
            }
        }

    except Exception as err:
        raise RuntimeError(f"Image processing failed: {err}")

# Serve uploaded or processed files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Default error page
@app.route('/error')
def error():
    return render_template('error.html', error="An error occurred during processing.")

# Run the Flask development server
if __name__ == '__main__':
    app.run(debug=True)
