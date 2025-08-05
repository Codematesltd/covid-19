import together
import os
import base64
from typing import List, Dict, Any
import json
import re

class LLMService:
    def __init__(self):
        # Initialize Together.ai client
        self.client = together.Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"  # Use available vision model
    
    def analyze_prediction(self, prediction: str, confidence: float, 
                          covid_prob: float, normal_prob: float, 
                          image_data: str, symptoms: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the CNN model's prediction using LLM with image analysis
        """
        try:
            # Prepare the prompt with image
            prompt = self._create_analysis_prompt(
                prediction, confidence, covid_prob, normal_prob, symptoms
            )
            
            # Prepare image for vision model
            image_url = f"data:image/jpeg;base64,{image_data}"
            
            # Call LLM with vision capabilities
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant. Analyze chest X-rays and provide JSON responses only. Always return valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            # Extract and clean response
            raw_response = response.choices[0].message.content
            print(f"Raw LLM response: {raw_response[:200]}...")  # Debug print
            
            # Parse JSON response with fallback
            analysis = self._parse_json_response(raw_response)
            
            return self._structure_response(analysis)
            
        except Exception as e:
            print(f"LLM analysis error: {str(e)}")
            return self._create_fallback_response(prediction, confidence)
    
    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON response with multiple fallback strategies"""
        try:
            # Try direct JSON parsing
            return json.loads(raw_response)
        except json.JSONDecodeError:
            try:
                # Extract JSON from markdown code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Extract JSON between curly braces
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                
                # If no JSON found, create structured response from text
                return self._extract_from_text(raw_response)
                
            except Exception as e:
                print(f"JSON parsing failed: {str(e)}")
                return self._extract_from_text(raw_response)
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text response"""
        lines = text.split('\n')
        result = {
            "explanation": "",
            "validation": "FLAGGED",
            "confidence_assessment": "MEDIUM",
            "medical_advice": "Please consult a healthcare professional.",
            "image_analysis": "",
            "key_indicators": [],
            "recommendations": []
        }
        
        # Extract key information from text
        for line in lines:
            line = line.strip()
            if "explanation" in line.lower():
                result["explanation"] = line.split(":", 1)[-1].strip()
            elif "validation" in line.lower():
                if "confirmed" in line.lower():
                    result["validation"] = "CONFIRMED"
            elif "confidence" in line.lower():
                if "high" in line.lower():
                    result["confidence_assessment"] = "HIGH"
                elif "low" in line.lower():
                    result["confidence_assessment"] = "LOW"
        
        # Use first part of text as explanation if none found
        if not result["explanation"]:
            result["explanation"] = text[:200] + "..." if len(text) > 200 else text
            
        return result
    
    def _create_fallback_response(self, prediction: str, confidence: float) -> Dict[str, Any]:
        """Create fallback response when LLM fails"""
        validation = "CONFIRMED" if confidence > 70 else "FLAGGED"
        conf_assessment = "HIGH" if confidence > 80 else "MEDIUM" if confidence > 60 else "LOW"
        
        return {
            "explanation": f"The model predicted {prediction} with {confidence:.1f}% confidence. LLM analysis unavailable.",
            "validation": validation,
            "confidence_assessment": conf_assessment,
            "medical_advice": "Please consult a healthcare professional for proper diagnosis and treatment.",
            "image_analysis": "Detailed image analysis is currently unavailable.",
            "key_indicators": ["Automated analysis pending"],
            "recommendations": ["Seek professional medical evaluation", "Consider additional imaging if symptoms persist"]
        }
    
    def _create_analysis_prompt(self, prediction: str, confidence: float, 
                               covid_prob: float, normal_prob: float, 
                               symptoms: List[str] = None) -> str:
        """Create structured prompt for LLM analysis"""
        
        symptoms_text = ", ".join(symptoms) if symptoms else "No symptoms provided"
        
        prompt = f"""
        Analyze this chest X-ray and CNN prediction. Respond with valid JSON only:

        MODEL PREDICTION: {prediction} ({confidence:.1f}% confidence)
        COVID PROBABILITY: {covid_prob:.1f}%
        NORMAL PROBABILITY: {normal_prob:.1f}%
        SYMPTOMS: {symptoms_text}

        Return JSON with these exact keys:
        {{
            "explanation": "Brief analysis of prediction and image",
            "validation": "CONFIRMED or FLAGGED",
            "confidence_assessment": "HIGH, MEDIUM, or LOW",
            "medical_advice": "Professional medical advice",
            "image_analysis": "What you see in the X-ray",
            "key_indicators": ["radiological findings"],
            "recommendations": ["medical recommendations"]
        }}

        Keep responses concise and professional.
        """
        
        return prompt
    
    def _structure_response(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Structure and validate the LLM response"""
        return {
            "explanation": analysis.get("explanation", "Analysis unavailable"),
            "validation": analysis.get("validation", "FLAGGED"),
            "confidence_assessment": analysis.get("confidence_assessment", "MEDIUM"),
            "medical_advice": analysis.get("medical_advice", "Consult a healthcare professional"),
            "image_analysis": analysis.get("image_analysis", "Image analysis unavailable"),
            "key_indicators": analysis.get("key_indicators", ["Analysis pending"]),
            "recommendations": analysis.get("recommendations", ["Seek professional medical evaluation"])
        }
