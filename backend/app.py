# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import base64
from PIL import Image
import io
import numpy as np
import cv2 # Make sure opencv-python is in your requirements.txt
import re
import random
import requests # Used for synchronous API calls
import json # Used for JSON parsing

# Explicitly import transforms components from torchvision to avoid conflicts
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# --- Configuration ---
# Define base directory of the project (one level up from 'backend')
# This is crucial for Flask to find the 'frontend' folder correctly when deployed
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# Initialize Flask app
# Tell Flask where to find static files (CSS, JS, images) and templates (HTML)
# Both are located in the 'frontend' directory relative to the project root
app = Flask(__name__,
            static_folder=FRONTEND_DIR,
            template_folder=FRONTEND_DIR)

# Enable CORS for all routes, allowing frontend (even if separate origin) to communicate
CORS(app) 

# --- Configuration for Gemini API ---
# In a real production environment, you would use a more robust secret management system.
# For local development or deployment environments, setting this as an environment variable
# before running the Flask app is the most common and secure approach.
Model_ADI_API_KEY = os.environ.get("GENERATIVE_LANGUAGE_API_KEY", "")
Model_ADI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

if not Model_ADI_API_KEY:
    print("Warning: GENERATIVE_LANGUAGE_API_KEY environment variable not set.")
    print("AI-powered features will be limited or unavailable.")

# --- U-Net Model Definition (from your original Streamlit code) ---
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Helper function to create a convolutional block
        def conv_block(in_c, out_c):
            return torch.nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder path
        self.enc1 = conv_block(3, 64)  # Input channels: 3 (RGB), Output: 64
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2) # Max pooling for downsampling

        # Decoder path (using ConvTranspose2d for upsampling)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128) # Concatenates with corresponding encoder output

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64) # Concatenates with corresponding encoder output

        self.final = nn.Conv2d(64, 1, kernel_size=1) # Final 1x1 convolution to output 1 channel (mask)

    def forward(self, x):
        # Encoder forward pass
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder forward pass with skip connections
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1) # Skip connection from e2
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # Skip connection from e1
        d1 = self.dec1(d1)

        out = self.final(d1) # Final output
        return out

# --- Set device and Load U-Net model (Loaded once globally when the app starts) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet()
try:
    # Ensure the model path is correct relative to the 'backend' directory
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'leaf_unet3_model.pth')
    # Load the trained model state dictionary
    unet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    unet.to(device).eval() # Set model to evaluation mode
    print("U-Net model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: U-Net model '{MODEL_PATH}' not found. Segmentation will not work.")
    unet = None # Set to None to indicate the model is not available
except Exception as e:
    print(f"ERROR loading U-Net model: {e}")
    unet = None

# --- Transforms for U-Net (Must match the transformations used during training) ---
# Now using the explicitly imported Compose, Resize, ToTensor, Normalize
transform_unet = Compose([
    Resize((256, 256)), # Resize images to 256x256
    ToTensor(),         # Convert PIL Image to PyTorch Tensor
    # Normalize with mean and standard deviation used during training
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper Functions (From your original Streamlit code, adapted for Flask context) ---

def create_circular_mask(height, width, center=None, radius=None):
    """
    Creates a circular binary mask for an image of given dimensions.
    Used to focus segmentation predictions within a central circular area.
    """
    if center is None:
        center = (int(width / 2), int(height / 2)) # Default to center of image
    if radius is None:
        radius = min(height, width) / 2 * 0.85 # Default radius based on smaller dimension

    Y, X = np.ogrid[:height, :width] # Create meshgrid coordinates
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2) # Calculate distance from center

    mask = dist_from_center <= radius # Pixels within radius are True
    return mask.astype(np.uint8) # Convert boolean mask to uint8 (0s and 1s)

def segment_with_unet(image_pil: Image.Image, mask_radius_factor=0.75):
    """
    Performs image segmentation using the pre-loaded U-Net model.
    Splits the image, applies a circular mask, and draws contours around detected regions.
    """
    if unet is None:
        # If U-Net model failed to load, return the original image as a fallback
        return np.array(image_pil) 

    original_width, original_height = image_pil.size # Store original image dimensions
    
    # Prepare image for U-Net: apply transformations and add batch dimension
    input_tensor = transform_unet(image_pil).unsqueeze(0).to(device)

    with torch.no_grad(): # Disable gradient calculation for inference
        output = unet(input_tensor) # Forward pass through U-Net
        # Apply sigmoid to output, remove batch dimension, convert to NumPy array
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()

    pred_h, pred_w = prediction.shape[0], prediction.shape[1] # Get dimensions of the prediction mask
    
    # Create and apply a circular mask to the prediction
    circular_mask_radius = int(min(pred_h, pred_w) / 2 * mask_radius_factor)
    circular_mask = create_circular_mask(pred_h, pred_w, radius=circular_mask_radius)
    masked_prediction = prediction * circular_mask # Apply the mask
    
    # Threshold the masked prediction to create a binary mask (0 or 255)
    masked_prediction = (masked_prediction > 0.5).astype(np.uint8) * 255

    # Convert PIL image to OpenCV format (BGR) for drawing contours
    img_np_resized = np.array(image_pil.resize((256, 256))) # Resize original for contour drawing base
    circled_img = cv2.cvtColor(img_np_resized, cv2.COLOR_RGB2BGR) # Convert RGB to BGR

    # Find contours in the binary mask
    contours, _ = cv2.findContours(masked_prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles around detected contours
    for cnt in contours:
        if cv2.contourArea(cnt) < 40: # Filter out small noise contours
            continue
        
        # Get minimum enclosing circle for each contour
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw the circle on the image (color BGR: Red (0, 0, 255), thickness 2)
        cv2.circle(circled_img, center, radius, (0, 0, 255), 2) 

    # Upscale the image with drawn circles back to the original resolution
    upscaled_circled_img = cv2.resize(circled_img, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    # Convert back to RGB for display (e.g., in web browser)
    return cv2.cvtColor(upscaled_circled_img, cv2.COLOR_BGR2RGB)

def format_remedies(text):
    """
    Formats raw text by normalizing spaces and inserting newlines after
    sentence endings (specifically, a letter followed by a period,
    avoiding decimals).
    """
    if not isinstance(text, str):
        text = str(text)

    # Replace all newline characters with a single space to flatten the text
    flattened_text = text.replace('\n', ' ')

    # Normalize all sequences of whitespace (multiple spaces) to a single space
    cleaned_text = re.sub(r'\s+', ' ', flattened_text).strip()

    result_lines = []
    current_line = []
    
    i = 0
    while i < len(cleaned_text):
        current_line.append(cleaned_text[i])

        # Check for a sentence boundary: a letter followed by a period
        if cleaned_text[i] == '.':
            if i > 0 and cleaned_text[i-1].isalpha(): 
                # Ensure it's not a decimal number (check if next character is a digit)
                is_decimal = (i + 1 < len(cleaned_text) and cleaned_text[i+1].isdigit())

                if not is_decimal:
                    # Look ahead to decide if a newline is appropriate.
                    # This checks if the period is followed by space and then an uppercase letter or digit,
                    # or if it's the very end of the string, implying a new sentence starts.
                    next_char_idx = i + 1
                    # Consume any spaces immediately after the period
                    while next_char_idx < len(cleaned_text) and cleaned_text[next_char_idx].isspace():
                        next_char_idx += 1 
                    
                    if next_char_idx >= len(cleaned_text) or \
                       cleaned_text[next_char_idx].isupper() or \
                       cleaned_text[next_char_idx].isdigit(): 
                        
                        result_lines.append("".join(current_line).strip()) # Add current sentence to lines
                        current_line = [] # Reset for next sentence
                        
                        # Adjust index to start after the consumed spaces for the next iteration
                        i = next_char_idx - 1 

        i += 1
    
    # Add any remaining text that didn't end with a sentence boundary
    if current_line:
        result_lines.append("".join(current_line).strip())

    # Join lines and remove any empty strings resulting from the process
    final_output = "\n".join(filter(None, result_lines))
    
    # Final cleanup: collapse multiple newlines to a single one and remove leading/trailing spaces on lines
    final_output = re.sub(r'\n\s*\n', '\n', final_output) 
    final_output = re.sub(r'\s+\n', '\n', final_output) 
    
    return final_output

# --- Helper Function for AI Interaction (Synchronous `requests` used within `async def` for clarity) ---
# Note: For production-grade async Flask, consider Quart or Flask-Async.
# This approach works but uses blocking `requests` calls inside `async def`
# which means the Flask worker will still be blocked during the API call.
def call_gemini_api_sync(prompt_parts, generation_config=None):
    if not Model_ADI_API_KEY:
        print("API key is missing. Cannot call Gemini API.")
        return {"error": "API Key Missing", "candidates": [{"content": {"parts": [{"text": "API Key Missing"}]}}]}

    url = f"{Model_ADI_API_URL}?key={Model_ADI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"role": "user", "parts": prompt_parts}]}
    
    if generation_config:
        payload["generationConfig"] = generation_config

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": f"API Call Failed: {str(e)}", "candidates": []}

def predict_disease_with_gemini(image_pil: Image.Image) -> str:
    """
    Sends a PIL Image to the Gemini API for disease prediction.
    Returns disease name in English and Bengali or an error message.
    """
    if not Model_ADI_API_KEY:
        return "API Key Missing" # Indicates a configuration error

    # Convert PIL Image to JPEG bytes, then base64 encode
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG") 
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    vision_prompt = """
    Analyze the provided plant or leaf image for health issues. Identify any abnormal conditions like diseases, pests, deficiencies, or decay.
    Respond STRICTLY with the exact name of the condition identified and the plant name, in English, followed by its Bengali translation in parentheses.
    Example: 'Tomato Early Blight (টমেটোর আগাম ধসা রোগ)'.
    If multiple conditions, prioritize the most severe.
    If no issues, reply 'Healthy'.
    If unrecognizable, reply 'I don't know'.
    If not a plant image, reply 'Not a plant image'.
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": vision_prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg", # Specify image type
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,    # Low temperature for deterministic output
            "maxOutputTokens": 100 # Max tokens for the response
        }
    }

    headers = {'Content-Type': 'application/json'}
    
    try:
        # Construct the URL, appending API key
        url = f"{Model_ADI_API_URL}?key={Model_ADI_API_KEY}"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        if result.get("candidates") and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content")
            if content and content.get("parts") and len(content["parts"]) > 0:
                disease_name = content["parts"][0].get("text", "Unknown").strip()
                return disease_name
        return "Failed to get disease name from Model_ADI." # Fallback if no valid prediction
    except requests.exceptions.RequestException as e:
        # Catch requests-specific errors (network, HTTP status)
        return f"Model_ADI Error: {e}"
    except json.JSONDecodeError:
        # Catch JSON parsing errors if response is not valid JSON
        return "Parse Error: Failed to parse Model_ADI response."
    except Exception as e:
        # Catch any other unexpected errors
        return f"Unexpected Error: {e}"

def get_remedies_with_gemini(disease_name: str, temp: float, humidity: float, ph: float, light: float, N: float, P: float, K: float, rain: str) -> dict:
    """
    Requests remedies and precautions from Gemini API based on disease and environmental factors.
    Uses structured output (JSON schema) for consistent responses.
    """
    if not Model_ADI_API_KEY:
        return {"Precautions": "API Key Missing", "Medicines": "API Key Missing"}

    # Convert rain status to a more descriptive string for the LLM
    rain_status_str = "it is raining or has rained recently" if rain == "1" else "it is not raining"

    remedy_prompt = f"""
    For the plant condition: '{disease_name}', considering the following environmental factors:
    Temperature: {temp}°C
    Humidity: {humidity}%
    Soil pH: {ph}
    Light Intensity: {light} Lux
    Soil Nutrients (NPK): Nitrogen {N}%, Phosphorus {P}%, Potassium {K}%
    Rain Status: {rain_status_str}

    Provide effective precautionary measures and suitable medicines/treatments.
    Format your response as a JSON object with two keys: "Precautions" and "Medicines".
    Each value should be a numbered list (1., 2., 3., etc.) of concise, single-sentence measures.
    Ensure both lists contain at least three relevant points.
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": remedy_prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json", # Request JSON output
            "responseSchema": { # Define the expected JSON structure
                "type": "OBJECT",
                "properties": {
                    "Precautions": {"type": "STRING"},
                    "Medicines": {"type": "STRING"}
                },
                "required": ["Precautions", "Medicines"] # Both keys are required
            },
            "temperature": 0.7,   # Higher temperature for more varied remedies
            "maxOutputTokens": 500 # More tokens for detailed remedies
        }
    }

    headers = {'Content-Type': 'application/json'}
    
    try:
        url = f"{Model_ADI_API_URL}?key={Model_ADI_API_KEY}"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        if result.get("candidates") and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content")
            if content and content.get("parts") and len(content["parts"]) > 0:
                # Parse the JSON string received from Gemini
                remedies_json = json.loads(content["parts"][0].get("text", "{}"))
                return {
                    "Precautions": remedies_json.get("Precautions", "No precautions found."),
                    "Medicines": remedies_json.get("Medicines", "No medicines found.")
                }
        return {"Precautions": "Failed to get remedies from AI.", "Medicines": "Failed to get remedies from AI."}
    except requests.exceptions.RequestException as e:
        return {"Precautions": f"API Error: {e}", "Medicines": f"API Error: {e}"}
    except json.JSONDecodeError:
        return {"Precautions": "Parse Error: Invalid JSON from AI.", "Medicines": "Parse Error: Invalid JSON from AI."}
    except Exception as e:
        return {"Precautions": f"Unexpected Error: {e}", "Medicines": f"Unexpected Error: {e}"}

def get_suitable_crops_with_gemini(temp: float, humidity: float, ph: float, light: float, N: float, P: float, K: float, rain: str) -> list[dict]:
    """
    Requests suitable crop recommendations from Gemini API based on environmental conditions.
    Returns a list of dictionaries, each with English and Bengali crop names.
    """
    if not Model_ADI_API_KEY:
        return [{"englishName": "API Key Missing", "bengaliName": "API কী অনুপস্থিত"}]

    rain_status_str = "it is raining or has rained recently" if rain == "1" else "it is not raining"

    # Prompt requests JSON array of objects
    crop_prompt = f"""
    Given the following environmental conditions:
    Temperature: {temp}°C
    Humidity: {humidity}%
    Soil pH: {ph}
    Light Intensity: {light} Lux
    Soil Nutrients (NPK): Nitrogen {N}%, Phosphorus {P}%, Potassium {K}%
    Rain Status: {rain_status_str}

    List up to 5 crops that would thrive in these conditions.
    Respond STRICTLY as a JSON array of objects, where each object has two keys: "englishName" and "bengaliName".
    Example: [{{ "englishName": "Wheat", "bengaliName": "গম" }}, {{ "englishName": "Rice", "bengaliName": "ধান" }}]
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": crop_prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json", # Request JSON array
            "responseSchema": { # Define the expected schema for array items
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "englishName": {"type": "STRING"},
                        "bengaliName": {"type": "STRING"}
                    },
                    "required": ["englishName", "bengaliName"] # Both names are required for each crop
                }
            },
            "temperature": 0.5,   # Moderate temperature for variety
            "maxOutputTokens": 200 # Sufficient tokens for a list of crops
        }
    }

    headers = {'Content-Type': 'application/json'}
    
    try:
        url = f"{Model_ADI_API_URL}?key={Model_ADI_API_KEY}"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        if result.get("candidates") and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content")
            if content and content.get("parts") and len(content["parts"]) > 0:
                crops_list = json.loads(content["parts"][0].get("text", "[]"))
                if isinstance(crops_list, list):
                    valid_crops = []
                    # Validate each item to ensure it matches the expected object structure
                    for item in crops_list:
                        if isinstance(item, dict) and "englishName" in item and "bengaliName" in item:
                            valid_crops.append(item)
                    return valid_crops
        return [{"englishName": "Failed to get crop recommendations from AI.", "bengaliName": "এআই থেকে ফসলের সুপারিশ পেতে ব্যর্থ হয়েছে।"}]
    except requests.exceptions.RequestException as e:
        return [{"englishName": f"API Error: {e}", "bengaliName": "এপিআই ত্রুটি"}]
    except json.JSONDecodeError:
        return [{"englishName": "Parse Error: Invalid JSON from AI.", "bengaliName": "পার্স ত্রুটি"}]
    except Exception as e:
        return [{"englishName": f"Unexpected Error: {e}", "bengaliName": "ত্রুটি"}]

# --- Flask Routes (API Endpoints) ---

# Route to serve the main HTML file (your single-page application)
@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

# Route to serve all other static files (CSS, JS, images, etc.)
# This catches requests like /style.css, /script.js, /Img_130716.png
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route('/recommend_crops', methods=['POST'])
def recommend_crops_endpoint(): # Changed function name to avoid conflict if any
    """
    API endpoint to receive environmental parameters and return suitable crop recommendations.
    """
    data = request.json # Get JSON data from the request body
    temp = data.get('temperature')
    humidity = data.get('humidity')
    ph = data.get('soil_ph')
    light = data.get('light_intensity')
    N = data.get('npk_n')
    P = data.get('npk_p')
    K = data.get('npk_k')
    rain = data.get('rain_status')

    # Basic validation for required parameters
    if None in [temp, humidity, ph, light, N, P, K, rain]:
        return jsonify({"error": "Missing one or more environmental parameters"}), 400

    # Call the helper function to get crop recommendations
    crops = get_suitable_crops_with_gemini(temp, humidity, ph, light, N, P, K, rain)
    return jsonify(crops) # Return the list of crops as JSON

@app.route('/analyze_plant', methods=['POST'])
def analyze_plant_endpoint(): # Changed function name
    """
    API endpoint to receive an image and environmental parameters for plant disease analysis,
    segmentation, and remedy suggestions.
    """
    data = request.json
    
    # Extract environmental parameters from the request data
    temp = data.get('temperature')
    humidity = data.get('humidity')
    ph = data.get('soil_ph')
    light = data.get('light_intensity')
    N = data.get('npk_n')
    P = data.get('npk_p')
    K = data.get('npk_k')
    rain = data.get('rain_status')
    image_b64 = data.get('image') # Base64 encoded image string

    # Validate that all necessary data is present
    if None in [temp, humidity, ph, light, N, P, K, rain, image_b64]:
        return jsonify({"error": "Missing image or environmental parameters"}), 400

    try:
        # Decode base64 image string into PIL Image object
        # The split(',') handles common data URI prefixes like "data:image/jpeg;base64,"
        image_bytes = base64.b64decode(image_b64.split(',')[1] if ',' in image_b64 else image_b64)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image format or decoding error: {e}"}), 400

    # --- AI Prediction (Disease Classification) ---
    predicted_disease = predict_disease_with_gemini(image_pil)
    
    # Handle direct errors returned by the prediction function
    if "API Key Missing" in predicted_disease or \
       "Model_ADI Error" in predicted_disease or \
       "Parse Error" in predicted_disease or \
       "Unexpected Error" in predicted_disease:
        # Return a server error if the prediction itself failed
        return jsonify({"error": predicted_disease}), 500
    
    # --- U-Net Segmentation ---
    # Perform segmentation and get the image with marked regions
    segmented_image_np = segment_with_unet(image_pil, mask_radius_factor=0.7)
    
    # Convert segmented image (NumPy array) back to PIL and then base64 for frontend display
    segmented_image_pil = Image.fromarray(segmented_image_np)
    buffered_segmented = io.BytesIO()
    # Save as PNG to preserve potential transparency or better quality for masks, though JPEG is fine too.
    segmented_image_pil.save(buffered_segmented, format="PNG") 
    segmented_image_b64 = base64.b64encode(buffered_segmented.getvalue()).decode('utf-8')

    # --- Generate Random Confidence ---
    # As per original Streamlit app, a random confidence is generated if a specific disease is predicted.
    ai_confidence = random.uniform(85.0, 99.0) 

    # --- Fetch Remedies and Precautions ---
    remedies_data = {"Precautions": "No recommendations.", "Medicines": "No recommendations."}
    # Only fetch remedies if a valid disease was predicted
    if predicted_disease not in ["I don't know", "Not a plant image", "Healthy", "Failed to get disease name from Model_ADI."]:
        remedies_data = get_remedies_with_gemini(
            predicted_disease, temp, humidity, ph, light, N, P, K, rain
        )
        # Apply the formatting helper to the retrieved remedies text
        remedies_data["Precautions"] = format_remedies(remedies_data["Precautions"])
        remedies_data["Medicines"] = format_remedies(remedies_data["Medicines"])
    
    # Return all results as a single JSON response
    return jsonify({
        "disease_name": predicted_disease,
        "confidence": round(ai_confidence, 2), # Round confidence for cleaner display
        "segmented_image": segmented_image_b64,
        "remedies": remedies_data
    })

# --- Main block to run the Flask app ---
if __name__ == '__main__':
    # You can change host and port as needed.
    # host='0.0.0.0' makes it accessible externally (useful for deployment/Docker).
    # debug=True enables debug mode (auto-reloads on code changes, provides debugger).
    app.run(debug=True, host='127.0.0.1', port=5000)
