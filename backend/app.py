import os
from flask import Flask, jsonify
from flask_cors import CORS
from backend.config import config
from backend.models.user import db

def create_app(config_name=None):
    """Application factory"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    from backend.routes.auth import auth_bp
    from backend.routes.predictions import prediction_bp
    from backend.routes.export import export_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(export_bp)
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'message': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'message': 'Internal server error'}), 500
    
    # Health check
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok', 'message': 'CVD Predictor API is running'}), 200
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
