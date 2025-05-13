import os
import logging
from flask import Flask
from routes import configure_routes

# Create the Flask app
app = Flask(__name__)

# Configurations
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

# Register routes
configure_routes(app)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
