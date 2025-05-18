#code for everything mixed:
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
from PIL import Image
import pytesseract
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
                'oriented': info["final"],
                'angle_info': info["angle_info"]
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


def add_subtle_outline(image):
    # """Adds a subtle outline around the detected document contour."""
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if contours:
    #     doc_contour = max(contours, key=cv2.contourArea)
    #     outlined_image = image.copy()
    #     cv2.drawContours(outlined_image, [doc_contour], -1, (32, 32, 33), )
    #     return outlined_image
    return image


def detect_and_correct_rotation(img):
    """Detects and corrects image rotation using Hough transform and OCR."""
    # Step 1: Convert to grayscale
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error converting to grayscale: {e}")
        return img, None

    # Step 2: Edge detection
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    except Exception as e:
        print(f"Error in edge detection: {e}")
        return img, None

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
        print("Tesseract not found. Make sure it's installed and in your system PATH.")
    except Exception as e:
        print(f"OCR orientation detection failed (continuing without it): {e}")

    angle_info = {
        "hough_angle": round(detected_angle, 2) if detected_angle is not None else None,
        "corrected_angle": round(corrected_angle, 2) if corrected_angle is not None else None,
        "ocr_angle": ocr_angle
    }
    
    return img, angle_info


def process_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        # Step 1: Load image
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError("Image file could not be read. Possibly corrupt or unsupported format.")

        # Step 2: Remove shadows and wrinkles using OpenCV
        cleaned = remove_shadow_and_wrinkles(img_cv)

        # Step 3: Save intermediate cleaned image
        cleaned_outlined = add_subtle_outline(cleaned)
        cleaned_rgb = cv2.cvtColor(cleaned_outlined, cv2.COLOR_BGR2RGB)
        cleaned_pil = Image.fromarray(cleaned_rgb).convert("RGBA")

        intermediate_filename = 'intermediate_' + os.path.splitext(os.path.basename(image_path))[0] + '.png'
        intermediate_path = os.path.join(os.path.dirname(image_path), intermediate_filename)
        cleaned_pil.save(intermediate_path, format='PNG')

        # Step 4: Remove background using U²-Net
        bg_removed_image = remove_bg_mult(cleaned_pil)
        
        # Save the background removed image before rotation
        bg_removed_np = np.array(bg_removed_image)
        bg_removed_cv = cv2.cvtColor(bg_removed_np, cv2.COLOR_RGBA2BGRA)
        
        # Step 5: Apply rotation correction to the background-removed image
        rotated_img, angle_info = detect_and_correct_rotation(cv2.cvtColor(bg_removed_cv, cv2.COLOR_BGRA2BGR))
        
        # Convert back to RGBA for final saving
        rotated_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
        rotated_pil = Image.fromarray(rotated_rgb).convert("RGBA")
        
        # Create alpha channel from the original background-removed image
        alpha_channel = bg_removed_image.split()[3]
        rotated_pil.putalpha(alpha_channel.resize(rotated_pil.size))
        
        # Save rotated image
        rotated_filename = 'rotated_' + os.path.basename(image_path)
        rotated_path = os.path.join(os.path.dirname(image_path), rotated_filename)
        cv2.imwrite(rotated_path, rotated_img)

        # Step 6: Save final output image
        final_filename = 'bg_removed_' + os.path.splitext(os.path.basename(image_path))[0] + '.png'
        final_path = os.path.join(os.path.dirname(image_path), final_filename)
        rotated_pil.save(final_path, format='PNG')

        return final_path, {
            "original": image_path,
            "intermediate": intermediate_path,
            "rotated": rotated_path,
            "final": final_path,
            "angle_info": angle_info,
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
            img_cv = cv2.imread(filepath)
            _, angle_info = detect_and_correct_rotation(img_cv)
            return jsonify({
                'success': True,
                'angle_info': angle_info
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})


if __name__ == '__main__':
    app.run(debug=True)
