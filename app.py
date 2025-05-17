import os
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
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
            processed_image_path, info = process_image(filepath)
            return render_template('result.html', result={
                'original': filepath,
                'oriented': processed_image_path
            })
        except Exception as e:
            return render_template('error.html', error=str(e))

    return redirect(request.url)


def process_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        # Step 1: Open input image
        input_image = Image.open(image_path).convert("RGBA")

        # Step 2: Apply background removal (U²-Net)
        output_image = remove_bg_mult(input_image)

        # Step 3: Save as PNG to support RGBA
        output_filename = 'bg_removed_' + os.path.splitext(os.path.basename(image_path))[0] + '.png'
        output_path = os.path.join(os.path.dirname(image_path), output_filename)
        output_image.save(output_path, format='PNG')

        return output_path, {"message": "Background removed successfully."}

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
