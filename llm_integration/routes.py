from flask import request, jsonify, current_app
from . import llm_bp
from .llm_service import LLMService
import base64
import io

@llm_bp.route('/analyze', methods=['POST'])
def analyze_with_llm():
    """
    Analyze prediction with LLM explanation and validation
    Expected JSON payload:
    {
        "prediction": "COVID" or "Normal",
        "confidence": 85.5,
        "covid_prob": 85.5,
        "normal_prob": 14.5,
        "image_data": "base64_encoded_image",
        "symptoms": ["cough", "fever"] (optional)
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['prediction', 'confidence', 'covid_prob', 'normal_prob', 'image_data']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Initialize LLM service
        llm_service = LLMService()
        
        # Get LLM analysis
        analysis = llm_service.analyze_prediction(
            prediction=data['prediction'],
            confidence=data['confidence'],
            covid_prob=data['covid_prob'],
            normal_prob=data['normal_prob'],
            image_data=data['image_data'],
            symptoms=data.get('symptoms', [])
        )
        
        return jsonify({
            "success": True,
            "original_prediction": {
                "prediction": data['prediction'],
                "confidence": data['confidence'],
                "covid_prob": data['covid_prob'],
                "normal_prob": data['normal_prob']
            },
            "llm_analysis": analysis
        })
        
    except Exception as e:
        return jsonify({"error": f"LLM analysis failed: {str(e)}"}), 500
