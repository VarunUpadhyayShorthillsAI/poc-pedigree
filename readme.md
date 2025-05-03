

This is a Flask-based web application that detects and corrects the orientation of uploaded document images using computer vision and OCR techniques. It supports image uploads and visually returns both the original and correctly oriented versions along with the detected rotation angles.

---

## 🚀 Features

* Upload `.jpg`, `.jpeg`, or `.png` images
* Detect document skew using:

  * **Edge detection + Hough Transform**
  * **Tesseract OCR-based orientation detection** (as a fallback)
* Automatically correct image orientation
* Returns the angle details:

  * Hough-detected angle
  * Corrected final angle
  * OCR-detected angle
* Offers both web UI and API (`/check_angle`) for angle checks
* Saves both original and corrected images for user reference

---

## ⚙️ Technologies Used

* Python + Flask
* OpenCV (`cv2`)
* NumPy
* Tesseract OCR (`pytesseract`)
* PIL (Python Imaging Library)
* HTML templates (Jinja2)

---

## 🖼️ Working Range

The orientation detection and correction logic performs **accurately and reliably** for document images within the following ranges:

✅ **Effective Working Angle Ranges:**

* **-45 50 45°**

⚠️ **Outside this range**, results might vary due to noise or ambiguous edge detection.



---

## 🧠 How It Works

1. **Edge Detection (Canny):** Highlights straight lines and document borders.
2. **Hough Transform:** Finds the skew angle based on the detected lines.
3. **Angle Normalization:** Ensures angles fall within a human-readable correction range.
4. **Rotation Matrix Application:** Corrects the skew based on detected angle.
5. **OCR Fallback:** Uses Tesseract OCR's OSD mode to refine or fix orientation if Hough detection fails.
6. **Result Rendering:** Outputs both original and corrected images along with metadata.

---

## 📂 Folder Structure

```
📁 uploads/           # Stores uploaded and corrected images
📄 app.py             # Main Flask server with logic
📄 templates/
   └── index.html     # Upload interface
   └── result.html    # Displays results
   └── error.html     # Handles error messages
```

---

## 📦 Setup Instructions

1. **Install Python packages**

   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract-OCR**

   * Windows: [Download here](https://github.com/tesseract-ocr/tesseract/wiki)
   * Linux:

     ```bash
     sudo apt install tesseract-ocr
     ```

3. **Run the Flask app**

   ```bash
   python app.py
   ```

4. **Navigate to**

   ```
   http://localhost:5000/
   ```

---

## 🧪 API Endpoint

**POST** `/check_angle`

> Use this endpoint to retrieve only the angle information (for AJAX or headless use).

### Request:

* Form-data with image file (`file`)

### Response:

```json
{
  "success": true,
  "angle_info": {
    "hough_angle": -12.0,
    "corrected_angle": -12.0,
    "ocr_angle": 0
  }
}
```

---
