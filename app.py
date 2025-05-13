import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import logging
from image_processing import process_image  # Import the new function

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Setup logger
app_logger = logging.getLogger('app_logger')
app_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
handler.setFormatter(formatter)
app_logger.addHandler(handler)

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
            rotated_image_path, angle_info = process_image(filepath, file)  # Call the new function
            return render_template('result.html', result={
                'original': filepath,
                'oriented': rotated_image_path,
                'angle_info': angle_info
            })
        except Exception as e:
            return render_template('error.html', error=str(e))

    return redirect(request.url)

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
            _, angle_info = process_image(filepath, file)  # Call the new function
            return jsonify({'success': True, 'angle_info': angle_info})
        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
