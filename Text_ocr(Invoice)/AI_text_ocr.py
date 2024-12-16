import os
import re
import cv2
import pytesseract
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fuzzywuzzy import process
from textblob import TextBlob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tesseract executable path (adjust to your system's location)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

app = FastAPI()

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

latest_excel_file_path = ""  # Global variable to store the latest file path
correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]

BRIGHTNESS = 1.0
GAMMA = 2.0
SATURATION = 2.0

def adjust_brightness_contrast(img, brightness=50, contrast=30):
    return cv2.convertScaleAbs(img, alpha=contrast / 127 + 1, beta=brightness - contrast)

def gamma_correction(img, gamma=1.5):
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, look_up_table)

def adjust_saturation(img, saturation=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    if avg_brightness < 100:
        img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
        img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
        img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
        _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_img

def correct_spelling_with_context(text):
    words = text.split()
    corrected_words = [str(TextBlob(word).correct()) for word in words]
    return " ".join(corrected_words)

def clean_text(extracted_text):
    unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
    lines = extracted_text.split('\n')
    processed_lines = []
    for line in lines:
        cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
        if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
            continue
        cleaned_line = correct_spelling_with_context(cleaned_line)
        numbers = ' '.join(re.findall(r'\d+', line))
        cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()
        if cleaned_line:
            processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
    return processed_lines

def fuzzy_match_keywords(text, correct_words):
    matched_keywords = []
    words_in_text = text.split()
    for word in words_in_text:
        best_match = process.extractOne(word, correct_words)
        if best_match[1] > 80:
            matched_keywords.append(best_match[0])
    return matched_keywords

def extract_text(image_path):
    preprocessed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(preprocessed_image)
    matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
    return clean_text(extracted_text), matched_keywords

def send_to_llm_api(cleaned_text):
    url = "http://192.168.100.75:3333/ask_anything/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        'query': (
            f'Please structure the following text into a clear table with these headers: '
            f'"{", ".join(correct_words)}". Output should follow markdown table format.\n\n'
            f'"{cleaned_text}"'
        )
    }

    try:
        logger.info("Sending request to LLM API...")
        response = requests.post(url, headers=headers, data=data, timeout=10)
        if response.status_code == 200:
            response_data = response.text
            logger.info("Received response from LLM API.")
            return response_data
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while connecting to the API: {e}")
        return None

def extract_table_from_text(text):
    match = re.search(r"(\|.*\|[\s\S]*\|.*\|)", text)
    if match:
        table_text = match.group(0)
        return table_text
    return None

def convert_table_to_dataframe(table_text):
    rows = table_text.split('\n')
    data = []
    columns = rows[0].split('|')[1:-1]  # Get columns from the first row (skip empty first and last part)

    for row in rows[1:]:
        cols = row.split('|')[1:-1]  # Split and clean each row, skip empty parts
        if len(cols) == len(columns):
            data.append([col.strip() for col in cols])

    df = pd.DataFrame(data, columns=columns)
    return df

# New function to clean and format extracted table into DataFrame
def fix_table_format(table_text):
    rows = table_text.split('\n')
    cleaned_data = []

    for row in rows:
        # Split columns by tab, space, or other delimiters, based on how the data is structured
        cols = [col.strip() for col in row.split('\t')]  # Adjust delimiter as necessary (e.g., "\t" for tabs)
        
        # Skip rows with no data or just empty spaces
        if len(cols) > 1 and any(cols):  # Ensure there is meaningful data in the row
            cleaned_data.append(cols)
    
    # Convert the cleaned data into a DataFrame
    df = pd.DataFrame(cleaned_data)
    return df

@app.post("/extract-text/")  # Endpoint for image processing
async def extract_text_from_image(file: UploadFile = File(...)):
    global latest_excel_file_path
    image_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    logger.info(f"Processing image: {image_path}")
    cleaned_text, matched_keywords = extract_text(image_path)
    
    logger.info("Sending cleaned text to LLM API for structuring.")
    structured_data = send_to_llm_api(" ".join(cleaned_text))

    if structured_data:
        table_text = extract_table_from_text(structured_data)
        if table_text:
            # Use fix_table_format function to clean and structure the table
            df = fix_table_format(table_text)
            latest_excel_file_path = os.path.join(UPLOAD_DIR, "Structured_Text_Extracted.xlsx")
            df.to_excel(latest_excel_file_path, index=False, engine='openpyxl')
            logger.info("Excel file with structured data created.")
        else:
            logger.error("No table found in the structured data.")

    return JSONResponse(content={
        "structured_data": structured_data if structured_data else "No structured data returned",
        "excel_file": latest_excel_file_path
    })

@app.get("/download-excel")  # Endpoint to download the generated Excel file
async def download_excel():
    if not latest_excel_file_path:
        return JSONResponse(content={"error": "No file found to download."}, status_code=404)
    return FileResponse(latest_excel_file_path, filename="Structured_Text_Extracted.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
