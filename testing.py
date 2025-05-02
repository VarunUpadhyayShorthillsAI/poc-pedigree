import pytesseract
from pytesseract import Output
import cv2

def get_document_orientation(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to ensure better resolution (if necessary)
    if image is not None:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Use Tesseract to detect orientation
    osd = pytesseract.image_to_osd(image, output_type=Output.DICT)

    rotation = osd.get("rotate", 0)
    orientation_confidence = osd.get("orientation_confidence", 0)

    print(f"Detected orientation angle: {rotation} degrees")
    print(f"Confidence: {orientation_confidence:.2f}")

    return rotation

# Example usage
get_document_orientation("rotated_315.jpeg")
