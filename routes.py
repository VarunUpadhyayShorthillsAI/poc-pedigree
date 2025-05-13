from flask import render_template, request, redirect, url_for, send_from_directory, jsonify
from image_processing import process_image
import os

# Helper function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to configure all routes
def configure_routes(app):
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
                rotated_image_path, angle_info = process_image(filepath, file)
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
                _, angle_info = process_image(filepath, file)
                return jsonify({'success': True, 'angle_info': angle_info})
            except Exception as e:
                return jsonify({'error': str(e)})

        return jsonify({'error': 'Invalid file type'})
