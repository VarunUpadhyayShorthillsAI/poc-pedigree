import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import logging

def setup_logger(name='image_processing_logger', level=logging.INFO):
    """Setup and return a configured logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handler if not already added to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def load_image(image_path, logger):
    """Load and convert image to grayscale"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    try:
        original_image = Image.open(image_path)
        grayscaled_image = original_image.convert("L")
        logger.info(f"Converted image to grayscale: {os.path.basename(image_path)}")
        return np.array(grayscaled_image)
    except Exception as e:
        logger.error(f"Failed to convert image to grayscale: {str(e)}")
        logger.warning(f"Using original image without grayscale conversion")
        return np.array(original_image)

def detect_angle_hough(img, logger):
    """Detect rotation angle using Hough Transform"""
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    detected_angle, corrected_angle = None, None
    
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
            detected_angle = bins[most_common_angle_idx] + (bins[1] - bins[0]) / 2
            logger.info(f"Detected document rotation angle: {detected_angle:.2f} degrees")

            corrected_angle = detected_angle
            if detected_angle < -45:
                corrected_angle += 180
            elif detected_angle > 45:
                corrected_angle -= 180

            logger.info(f"Corrected angle after normalization: {corrected_angle:.2f} degrees")
    
    return detected_angle, corrected_angle

def rotate_image(img, angle, logger):
    """Rotate image by given angle"""
    if angle is None:
        return img
        
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def detect_ocr_orientation(img, logger):
    """Detect image orientation using OCR"""
    try:
        osd = pytesseract.image_to_osd(Image.fromarray(img), output_type=pytesseract.Output.DICT)
        ocr_angle = osd.get("rotate", 0)
        if ocr_angle not in [0, None]:
            logger.info(f"OCR detected orientation adjustment: {ocr_angle} degrees")
            return ocr_angle
        return None
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError("Tesseract not found. Make sure it's installed and in your system PATH.")
    except Exception as e:
        logger.warning(f"OCR orientation detection failed (continuing without it): {e}")
        return None

def save_image(img, original_path, logger):
    """Save processed image"""
    rotated_image_path = os.path.join(os.path.dirname(original_path), 
                               'rotated_' + os.path.basename(original_path))
    cv2.imwrite(rotated_image_path, img)
    return rotated_image_path

def process_image(image_path, file=None):
    """Main function to process and correct image orientation"""
    logger = setup_logger()
    
    try:
        # Use filename from file object if available, otherwise use image_path
        filename = file.filename if file and hasattr(file, 'filename') else os.path.basename(image_path)
        
        # Load and preprocess image
        img = load_image(image_path, logger)
        if img is None:
            raise ValueError("Image file could not be read. Possibly corrupt or unsupported format.")
            
        # Detect angle using Hough Transform
        detected_angle, corrected_angle = detect_angle_hough(img, logger)
        
        # Apply Hough rotation if detected
        if corrected_angle is not None:
            img = rotate_image(img, corrected_angle, logger)
            
        # Detect and apply OCR-based orientation
        ocr_angle = detect_ocr_orientation(img, logger)
        if ocr_angle is not None and abs(ocr_angle) >= 20:
            img = rotate_image(img, -ocr_angle, logger)
            
        # Save result
        rotated_image_path = save_image(img, image_path, logger)
        
        # Return results
        angle_info = {
            "hough_angle": round(detected_angle, 2) if detected_angle is not None else None,
            "corrected_angle": round(corrected_angle, 2) if corrected_angle is not None else None,
            "ocr_angle": ocr_angle
        }
        
        return rotated_image_path, angle_info

    except Exception as err:
        raise RuntimeError(f"Image processing failed: {err}")

# Example usage:
# rotated_path, angle_info = process_image("path/to/image.jpg")