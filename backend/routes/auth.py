from flask import Blueprint, request, jsonify
from backend.models.user import db, User
from backend.utils.auth import generate_token, token_required
from sqlalchemy import or_

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    # Validate input
    if not data.get('email') or not data.get('password') or not data.get('username'):
        return jsonify({'message': 'Email, username, and password are required'}), 400
    
    # Check if user already exists
    if User.query.filter(or_(User.email == data['email'], User.username == data['username'])).first():
        return jsonify({'message': 'Email or username already exists'}), 409
    
    # Create new user
    user = User(
        username=data['username'],
        email=data['email'],
        first_name=data.get('first_name', ''),
        last_name=data.get('last_name', ''),
        language=data.get('language', 'en')
    )
    user.set_password(data['password'])
    
    try:
        db.session.add(user)
        db.session.commit()
        
        # Generate token
        token = generate_token(user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': user.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error registering user: {str(e)}'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    if not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Email and password are required'}), 400
    
    # Find user
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'message': 'Invalid email or password'}), 401
    
    if not user.is_active:
        return jsonify({'message': 'User account is inactive'}), 403
    
    # Generate token
    token = generate_token(user.id)
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict()
    }), 200

@auth_bp.route('/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    """Get current user profile"""
    return jsonify(current_user.to_dict()), 200

@auth_bp.route('/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    """Update user profile"""
    data = request.get_json()
    
    current_user.first_name = data.get('first_name', current_user.first_name)
    current_user.last_name = data.get('last_name', current_user.last_name)
    current_user.age = data.get('age', current_user.age)
    current_user.gender = data.get('gender', current_user.gender)
    current_user.language = data.get('language', current_user.language)
    
    try:
        db.session.commit()
        return jsonify({
            'message': 'Profile updated successfully',
            'user': current_user.to_dict()
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error updating profile: {str(e)}'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@token_required
def change_password(current_user):
    """Change user password"""
    data = request.get_json()
    
    if not data.get('old_password') or not data.get('new_password'):
        return jsonify({'message': 'Old and new passwords are required'}), 400
    
    if not current_user.check_password(data['old_password']):
        return jsonify({'message': 'Current password is incorrect'}), 401
    
    current_user.set_password(data['new_password'])
    
    try:
        db.session.commit()
        return jsonify({'message': 'Password changed successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error changing password: {str(e)}'}), 500

@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout(current_user):
    """Logout user (client-side token deletion)"""
    return jsonify({'message': 'Logged out successfully'}), 200
