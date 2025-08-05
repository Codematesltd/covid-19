import base64
import io
from PIL import Image
import numpy as np

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def prepare_image_for_llm(image, max_size=(512, 512)):
    """Prepare image for LLM analysis by resizing and optimizing"""
    # Resize image to reduce token usage
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image
