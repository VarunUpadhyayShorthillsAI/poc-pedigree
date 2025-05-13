import os
from flask import Flask

def create_app():
    app = Flask(__name__, template_folder='../templates')
    
    # Ensure the uploads directory exists
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Register blueprints
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app