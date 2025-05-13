import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import logging

# Setup logger for image processing
image_logger = logging.getLogger('image_processing_logger')
image_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
handler.setFormatter(formatter)
image_logger.addHandler(handler)

def process_image(image_path, file):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        try:
            original_image = Image.open(image_path)
            grayscaled_image = original_image.convert("L")
            image_logger.info(f"Converted image to grayscale: {file.filename}")
        except Exception as e:
            image_logger.error(f"Failed to convert image to grayscale: {str(e)}")
            image_logger.warning(f"Using original image without grayscale conversion")
            grayscaled_image = original_image

        img = np.array(grayscaled_image)

        if img is None:
            raise ValueError("Image file could not be read. Possibly corrupt or unsupported format.")

        # Edge detection using Canny
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        # Hough Transform
        detected_angle, corrected_angle = None, None
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
                detected_angle = bins[most_common_angle_idx] + (bins[1] - bins[0]) / 2
                image_logger.info(f"Detected document rotation angle: {detected_angle:.2f} degrees")

                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)

                corrected_angle = detected_angle
                if detected_angle < -45:
                    corrected_angle += 180
                elif detected_angle > 45:
                    corrected_angle -= 180

                image_logger.info(f"Corrected angle after normalization: {corrected_angle:.2f} degrees")

                M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # OCR Orientation detection
        ocr_angle = None
        try:
            osd = pytesseract.image_to_osd(Image.fromarray(img), output_type=pytesseract.Output.DICT)
            ocr_angle = osd.get("rotate", 0)
            if ocr_angle not in [0, None]:
                image_logger.info(f"OCR detected orientation adjustment: {ocr_angle} degrees")
                if abs(ocr_angle) >= 5:
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, -ocr_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract not found. Make sure it's installed and in your system PATH.")
        except Exception as e:
            image_logger.warning(f"OCR orientation detection failed (continuing without it): {e}")

        # Save result
        angle_info = {
            "hough_angle": round(detected_angle, 2) if detected_angle is not None else None,
            "corrected_angle": round(corrected_angle, 2) if corrected_angle is not None else None,
            "ocr_angle": ocr_angle
        }

        rotated_image_path = os.path.join(os.path.dirname(image_path), 'rotated_' + os.path.basename(image_path))
        cv2.imwrite(rotated_image_path, img)
        return rotated_image_path, angle_info

    except Exception as err:
        raise RuntimeError(f"Image processing failed: {err}")
