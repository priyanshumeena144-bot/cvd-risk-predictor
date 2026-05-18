import jwt
from functools import wraps
from flask import request, jsonify, current_app
from datetime import datetime, timedelta
from backend.models.user import User

def generate_token(user_id):
    """Generate JWT token"""
    payload = {
        'user_id': user_id,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + current_app.config['JWT_ACCESS_TOKEN_EXPIRES']
    }
    return jwt.encode(
        payload,
        current_app.config['JWT_SECRET_KEY'],
        algorithm='HS256'
    )

def verify_token(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(
            token,
            current_app.config['JWT_SECRET_KEY'],
            algorithms=['HS256']
        )
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check for token in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        payload = verify_token(token)
        if not payload:
            return jsonify({'message': 'Invalid or expired token'}), 401
        
        current_user = User.query.get(payload['user_id'])
        if not current_user or not current_user.is_active:
            return jsonify({'message': 'User not found or inactive'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated
