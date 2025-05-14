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
    """Load image and preprocess it to improve OCR & orientation correction"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    try:
        original_image = cv2.imread(image_path)

        if original_image is None:
            raise ValueError("cv2.imread() failed to read the image.")
        
        logger.info(f"Loaded image: {os.path.basename(image_path)}")

        # Step 1: Safe shadow correction using illumination normalization
        # gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        # bg = cv2.medianBlur(gray, 21)
        # normalized = cv2.divide(gray, bg, scale=255)
        # logger.info("Illumination normalization (safe shadow correction) applied.")

        # Step 2: White balance correction
        wb = cv2.xphoto.createSimpleWB()
        white_balanced = wb.balanceWhite(original_image)
        logger.info("White balance correction applied.")

        # Step 3: Perspective correction (optional)
        blurred = cv2.GaussianBlur(white_balanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if contours:
            page = contours[0]
            epsilon = 0.02 * cv2.arcLength(page, True)
            approx = cv2.approxPolyDP(page, epsilon, True)
            if len(approx) == 4:
                pts = np.float32([pt[0] for pt in approx])
                rect = np.array(sorted(pts, key=lambda x: (x[1], x[0])))
                (tl, tr, br, bl) = rect
                width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
                height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
                dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(white_balanced, M, (width, height))
                logger.info("Perspective correction applied.")
            else:
                warped = white_balanced
                logger.info("Could not find 4 corners for perspective correction.")
        else:
            warped = white_balanced
            logger.info("No contours found for perspective correction.")

        # Step 4: Final grayscale conversion
        grayscaled_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        logger.info("Converted image to grayscale.")
        return grayscaled_image

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise RuntimeError(f"Preprocessing failed: {str(e)}")



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
        if ocr_angle is not None and abs(ocr_angle) >= 40:
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