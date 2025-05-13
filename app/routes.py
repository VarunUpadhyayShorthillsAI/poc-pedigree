import os
from flask import Blueprint, render_template, request, redirect, jsonify, send_from_directory, current_app
from .utils import allowed_file
from .image_processing import process_image

main = Blueprint('main', __name__)
UPLOAD_FOLDER = 'uploads'

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        # Ensure uploads directory exists
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
            
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            rotated_image_path, angle_info = process_image(filepath, file)
            
            # Get just the filenames for display in template
            original_filename = os.path.basename(filepath)
            rotated_filename = os.path.basename(rotated_image_path)
            
            return render_template('result.html', result={
                'original': original_filename,
                'oriented': rotated_filename,
                'angle_info': angle_info
            })
        except Exception as e:
            return render_template('error.html', error=str(e))
    return redirect(request.url)

# This route serves files from the uploads directory
@main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@main.route('/error')
def error():
    return render_template('error.html', error="An error occurred.")

@main.route('/check_angle', methods=['POST'])
def check_angle():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file selected'})

    file = request.files['file']
    if file and allowed_file(file.filename):
        # Ensure uploads directory exists
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
            
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        try:
            _, angle_info = process_image(filepath, file)
            return jsonify({'success': True, 'angle_info': angle_info})
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'error': 'Invalid file type'})