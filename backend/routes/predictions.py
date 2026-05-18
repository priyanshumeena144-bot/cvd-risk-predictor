from flask import Blueprint, request, jsonify
from backend.models.user import db, HealthPrediction
from backend.utils.auth import token_required
from backend.utils.prediction import PredictionEngine

prediction_bp = Blueprint('predictions', __name__, url_prefix='/api/predictions')
prediction_engine = PredictionEngine()

@prediction_bp.route('/predict', methods=['POST'])
@token_required
def make_prediction(current_user):
    """Make a CVD risk prediction"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'bmi', 'glucose', 'heart_rate']
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing required health metrics'}), 400
    
    # Make prediction
    prediction_result = prediction_engine.predict(data)
    
    if prediction_result['error']:
        return jsonify({'message': prediction_result['error']}), 500
    
    # Generate recommendations
    recommendations = prediction_engine.get_recommendations(data, prediction_result['risk_category'])
    
    # Save to database
    health_prediction = HealthPrediction(
        user_id=current_user.id,
        age=data.get('age'),
        gender=data.get('gender'),
        cigarettes_per_day=data.get('cigarettes_per_day', 0),
        systolic_bp=data.get('systolic_bp'),
        diastolic_bp=data.get('diastolic_bp'),
        cholesterol=data.get('cholesterol'),
        bmi=data.get('bmi'),
        glucose=data.get('glucose'),
        heart_rate=data.get('heart_rate'),
        stroke=data.get('stroke', False),
        hypertension=data.get('hypertension', False),
        diabetes=data.get('diabetes', False),
        current_smoker=data.get('current_smoker', False),
        prevalent_stroke=data.get('prevalent_stroke', False),
        prevalent_hyp=data.get('prevalent_hyp', False),
        diabetes_status=data.get('diabetes_status'),
        education=data.get('education'),
        risk_score=prediction_result['risk_score'],
        risk_percentage=prediction_result['risk_percentage'],
        risk_category=prediction_result['risk_category'],
        recommendations='\n'.join(recommendations),
        notes=data.get('notes', '')
    )
    
    try:
        db.session.add(health_prediction)
        db.session.commit()
        
        return jsonify({
            'message': 'Prediction made successfully',
            'prediction': {
                'id': health_prediction.id,
                'risk_score': health_prediction.risk_score,
                'risk_percentage': health_prediction.risk_percentage,
                'risk_category': health_prediction.risk_category,
                'recommendations': recommendations,
                'created_at': health_prediction.created_at.isoformat()
            }
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error saving prediction: {str(e)}'}), 500

@prediction_bp.route('/history', methods=['GET'])
@token_required
def get_prediction_history(current_user):
    """Get user's prediction history"""
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    predictions = HealthPrediction.query.filter_by(user_id=current_user.id)\
        .order_by(HealthPrediction.created_at.desc())\
        .limit(limit)\
        .offset(offset)\
        .all()
    
    total = HealthPrediction.query.filter_by(user_id=current_user.id).count()
    
    return jsonify({
        'predictions': [p.to_dict() for p in predictions],
        'total': total,
        'limit': limit,
        'offset': offset
    }), 200

@prediction_bp.route('/<int:prediction_id>', methods=['GET'])
@token_required
def get_prediction(current_user, prediction_id):
    """Get a specific prediction"""
    prediction = HealthPrediction.query.filter_by(
        id=prediction_id,
        user_id=current_user.id
    ).first()
    
    if not prediction:
        return jsonify({'message': 'Prediction not found'}), 404
    
    return jsonify(prediction.to_dict()), 200

@prediction_bp.route('/<int:prediction_id>', methods=['DELETE'])
@token_required
def delete_prediction(current_user, prediction_id):
    """Delete a prediction"""
    prediction = HealthPrediction.query.filter_by(
        id=prediction_id,
        user_id=current_user.id
    ).first()
    
    if not prediction:
        return jsonify({'message': 'Prediction not found'}), 404
    
    try:
        db.session.delete(prediction)
        db.session.commit()
        return jsonify({'message': 'Prediction deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error deleting prediction: {str(e)}'}), 500

@prediction_bp.route('/compare', methods=['POST'])
@token_required
def compare_predictions(current_user):
    """Compare multiple predictions"""
    data = request.get_json()
    prediction_ids = data.get('prediction_ids', [])
    
    if len(prediction_ids) < 2:
        return jsonify({'message': 'At least 2 predictions required for comparison'}), 400
    
    predictions = HealthPrediction.query.filter(
        HealthPrediction.id.in_(prediction_ids),
        HealthPrediction.user_id == current_user.id
    ).all()
    
    if len(predictions) < 2:
        return jsonify({'message': 'Not enough valid predictions found'}), 404
    
    # Calculate trends
    sorted_preds = sorted(predictions, key=lambda x: x.created_at)
    risk_trend = [{'date': p.created_at.isoformat(), 'percentage': p.risk_percentage} for p in sorted_preds]
    
    return jsonify({
        'predictions': [p.to_dict() for p in predictions],
        'risk_trend': risk_trend,
        'improvement': sorted_preds[-1].risk_percentage - sorted_preds[0].risk_percentage
    }), 200
