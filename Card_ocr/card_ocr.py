import os
import easyocr
import requests
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import re
from PIL import Image
import io
import numpy as np

# FastAPI app initialization
app = FastAPI()

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic model to define the structure of the card data
class CardDetails(BaseModel):
    username: str
    expiry_date: str
    card_number: str
    card_type: str

# Function to extract details from text using the external API
def extract_card_details_from_text(text):
    # API endpoint and headers
    url = "http://192.168.100.75:3333/ask_anything/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    # Updated data to be sent in the request
    data = {
        'query': f'Please extract the following details from this text: "{text}".\n'
                '1. username\n'
                '2. expiry date (MM/YY format)\n'
                '3. card number (XXXX XXXX XXXX XXXX format)\n'
                '4. card type (e.g., Visa, MasterCard, etc.)\n'
                'Please return the values with their respective headings like this:\n'
                'username: [value]\nexpiry date: [value]\ncard number: [value]\ncard type: [value]\n'
                'Do not include any extra symbols like stars or quotation marks. Just the values.'
    }
    
    # Send request and handle response
    response = requests.post(url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}, {response.text}")
    
    return response.text

# Function to extract raw text from card image using easyocr with resizing
def extract_raw_text(image_bytes):
    # Open the image using Pillow
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize the image (example: resize to width=800px, maintaining aspect ratio)
    base_width = 800
    w_percent = base_width / float(image.size[0])
    h_size = int((float(image.size[1]) * float(w_percent)))
    
    # Use LANCZOS for high-quality downsampling (replacing the old ANTI_ALIAS)
    image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
    
    # Convert the resized image to a numpy array (easyocr accepts numpy arrays)
    image_array = np.array(image)

    # Initialize the OCR reader (you can add more languages if needed)
    reader = easyocr.Reader(['en'])
    
    # Perform OCR on the resized image (passing the numpy array)
    result = reader.readtext(image_array)
    
    # Extract text from the OCR result (easyocr returns a list of tuples)
    raw_text = [text[1] for text in result]
    return raw_text

# Function to process the card image and extract details
def process_card_image(image_file):
    # Read the uploaded image file as bytes
    image_data = image_file.file.read()

    # Extract raw text from the card image
    raw_text_array = extract_raw_text(image_data)
    
    # Combine all text into a single string
    combined_text = " ".join(raw_text_array)
    
    # Query the API for details
    try:
        card_details_response = extract_card_details_from_text(combined_text)
    except Exception as e:
        return {"error": str(e)}  # Return error as a dictionary
    
    # Initialize structured data to be returned
    structured_data = {
        "username": "N/A",
        "expiry_date": "N/A",
        "card_number": "N/A",
        "card_type": "N/A"
    }

    # Clean the raw response and extract the relevant details using regular expressions
    card_details_response = card_details_response.replace("\\n", " ")  # Remove newline characters

    # Use regular expressions to extract values from the response string
    username_match = re.search(r"username:\s*([A-Za-z\s]+?)(?=\s*expiry date)", card_details_response)
    expiry_date_match = re.search(r"expiry date:\s*(\d{2}/\d{2})", card_details_response)
    card_number_match = re.search(r"card number:\s*(\d{4} \d{4} \d{4} \d{4})", card_details_response)
    card_type_match = re.search(r"card type:\s*([A-Za-z\s]+)", card_details_response)

    # Update the structured data dictionary if matches are found
    if username_match:
        structured_data["username"] = username_match.group(1).strip()
    if expiry_date_match:
        structured_data["expiry_date"] = expiry_date_match.group(1).strip()
    if card_number_match:
        structured_data["card_number"] = card_number_match.group(1).strip()
    if card_type_match:
        structured_data["card_type"] = card_type_match.group(1).strip()
    
    # If all fields are still "N/A", suggest a clearer image
    if (structured_data["username"] == "N/A" and 
        structured_data["expiry_date"] == "N/A" and
        structured_data["card_number"] == "N/A" and
        structured_data["card_type"] == "N/A"):
        return {
            "error": "Unable to extract details. Please upload a clearer image."
        }

    # Return the structured data
    return structured_data

# FastAPI endpoint to upload and process the card image
@app.post("/extract-card-details/")
async def get_card_details(image_file: UploadFile = File(...)):
    # Process the uploaded image and extract the details
    details = process_card_image(image_file)
    return details
