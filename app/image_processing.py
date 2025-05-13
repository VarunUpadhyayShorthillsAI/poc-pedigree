import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger('app_logger')

def process_image(image_path, file):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    try:
        original_image = Image.open(image_path)
        grayscaled_image = original_image.convert("L")
        logger.info(f"Converted image to grayscale: {file.filename}")
    except Exception as e:
        logger.error(f"Failed to convert image to grayscale: {str(e)}")
        logger.warning("Using original image without grayscale conversion")
        grayscaled_image = original_image

    img = np.array(grayscaled_image)
    if img is None:
        raise ValueError("Image could not be read.")

    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    detected_angle = None
    corrected_angle = None

    if lines is not None and len(lines) > 0:
        angles = [np.degrees(theta) - 90 for rho, theta in lines[:, 0]]
        angles = [(a + 180 if a < -90 else a - 180 if a > 90 else a) for a in angles]
        hist, bins = np.histogram(angles, bins=180, range=(-90, 90))
        detected_angle = bins[np.argmax(hist)] + (bins[1] - bins[0]) / 2
        logger.info(f"Detected angle: {detected_angle:.2f}°")

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        corrected_angle = detected_angle
        if detected_angle < -45:
            corrected_angle += 180
        elif detected_angle > 45:
            corrected_angle -= 180

        logger.info(f"Corrected angle: {corrected_angle:.2f}°")
        M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    ocr_angle = None
    try:
        osd = pytesseract.image_to_osd(Image.fromarray(img), output_type=pytesseract.Output.DICT)
        ocr_angle = osd.get("rotate", 0)
        if abs(ocr_angle) >= 5:
            M = cv2.getRotationMatrix2D(center, -ocr_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            logger.info(f"OCR angle corrected: {ocr_angle}°")
    except Exception as e:
        logger.warning(f"OCR orientation failed: {e}")

    angle_info = {
        "hough_angle": round(detected_angle, 2) if detected_angle else None,
        "corrected_angle": round(corrected_angle, 2) if corrected_angle else None,
        "ocr_angle": ocr_angle
    }

    rotated_image_path = os.path.join(os.path.dirname(image_path), 'rotated_' + os.path.basename(image_path))
    cv2.imwrite(rotated_image_path, img)
    return rotated_image_path, angle_info
