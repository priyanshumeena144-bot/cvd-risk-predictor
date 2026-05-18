from flask import Blueprint, request, jsonify, send_file
from backend.models.user import HealthPrediction
from backend.utils.auth import token_required
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import csv
import io
import os

export_bp = Blueprint('export', __name__, url_prefix='/api/export')

@export_bp.route('/pdf/<int:prediction_id>', methods=['GET'])
@token_required
def export_pdf(current_user, prediction_id):
    """Export prediction as PDF"""
    prediction = HealthPrediction.query.filter_by(
        id=prediction_id,
        user_id=current_user.id
    ).first()
    
    if not prediction:
        return jsonify({'message': 'Prediction not found'}), 404
    
    try:
        # Create PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#e74c3c'),
            spaceAfter=30,
            alignment=1
        )
        elements.append(Paragraph("❤️ CVD Risk Assessment Report", title_style))
        
        # Date and patient info
        info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=10)
        elements.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", info_style))
        elements.append(Paragraph(f"<b>Patient:</b> {current_user.first_name} {current_user.last_name}", info_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Risk Summary
        risk_color = {
            'low': '#2ecc71',
            'medium': '#f39c12',
            'high': '#e74c3c'
        }.get(prediction.risk_category, '#3498db')
        
        summary_style = ParagraphStyle(
            'Summary',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor(risk_color),
            spaceAfter=10
        )
        elements.append(Paragraph(f"Risk Level: {prediction.risk_category.upper()}", summary_style))
        elements.append(Paragraph(f"10-Year CHD Risk: {prediction.risk_percentage}%", summary_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Health Metrics
        elements.append(Paragraph("Health Metrics", styles['Heading3']))
        metrics_data = [
            ['Metric', 'Value'],
            ['Age', str(prediction.age)],
            ['Gender', prediction.gender or 'N/A'],
            ['Systolic BP', f"{prediction.systolic_bp} mmHg"],
            ['Diastolic BP', f"{prediction.diastolic_bp} mmHg"],
            ['Cholesterol', f"{prediction.cholesterol} mg/dL"],
            ['BMI', f"{prediction.bmi}"],
            ['Glucose', f"{prediction.glucose} mg/dL"],
            ['Heart Rate', f"{prediction.heart_rate} bpm"],
        ]
        
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Medical History
        elements.append(Paragraph("Medical History", styles['Heading3']))
        history_data = [
            ['Condition', 'Status'],
            ['Stroke', 'Yes' if prediction.stroke else 'No'],
            ['Hypertension', 'Yes' if prediction.hypertension else 'No'],
            ['Diabetes', 'Yes' if prediction.diabetes else 'No'],
            ['Current Smoker', 'Yes' if prediction.current_smoker else 'No'],
        ]
        
        history_table = Table(history_data)
        history_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(history_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        if prediction.recommendations:
            elements.append(Paragraph("Recommendations", styles['Heading3']))
            for rec in prediction.recommendations.split('\n'):
                elements.append(Paragraph(f"• {rec}", styles['Normal']))
        
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(
            "⚠️ Disclaimer: This report is for informational purposes only. Please consult with a healthcare professional for medical advice.",
            ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.red)
        ))
        
        # Build PDF
        doc.build(elements)
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'CVD_Report_{prediction_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    
    except Exception as e:
        return jsonify({'message': f'Error generating PDF: {str(e)}'}), 500

@export_bp.route('/csv/history', methods=['GET'])
@token_required
def export_history_csv(current_user):
    """Export prediction history as CSV"""
    try:
        predictions = HealthPrediction.query.filter_by(user_id=current_user.id)\
            .order_by(HealthPrediction.created_at.asc()).all()
        
        if not predictions:
            return jsonify({'message': 'No predictions to export'}), 404
        
        # Create CSV
        csv_buffer = io.StringIO()
        fieldnames = [
            'Date', 'Age', 'Gender', 'Systolic BP', 'Diastolic BP',
            'Cholesterol', 'BMI', 'Glucose', 'Heart Rate',
            'Stroke', 'Hypertension', 'Diabetes', 'Current Smoker',
            'Risk Score', 'Risk Percentage', 'Risk Category'
        ]
        
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        
        for pred in predictions:
            writer.writerow({
                'Date': pred.created_at.strftime('%Y-%m-%d %H:%M'),
                'Age': pred.age,
                'Gender': pred.gender or '',
                'Systolic BP': pred.systolic_bp,
                'Diastolic BP': pred.diastolic_bp,
                'Cholesterol': pred.cholesterol,
                'BMI': pred.bmi,
                'Glucose': pred.glucose,
                'Heart Rate': pred.heart_rate,
                'Stroke': pred.stroke,
                'Hypertension': pred.hypertension,
                'Diabetes': pred.diabetes,
                'Current Smoker': pred.current_smoker,
                'Risk Score': pred.risk_score,
                'Risk Percentage': pred.risk_percentage,
                'Risk Category': pred.risk_category
            })
        
        csv_buffer.seek(0)
        byte_buffer = io.BytesIO(csv_buffer.getvalue().encode())
        
        return send_file(
            byte_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'CVD_History_{current_user.id}_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    
    except Exception as e:
        return jsonify({'message': f'Error generating CSV: {str(e)}'}), 500
