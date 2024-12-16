# # from fastapi import FastAPI, File, UploadFile
# # from fastapi.responses import JSONResponse, FileResponse
# # import cv2
# # import pytesseract
# # import re
# # import pandas as pd
# # import os

# # # Set the path to the Tesseract executable
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# # app = FastAPI()

# # # Path to save the uploaded image
# # UPLOAD_DIR = "uploads/"
# # os.makedirs(UPLOAD_DIR, exist_ok=True)

# # # Path to save the generated Excel file
# # EXCEL_FILE_PATH = "Final_extracted_data.xlsx"

# # def extract_text(image_path):
# #     # Read the image using OpenCV
# #     img = cv2.imread(image_path)
    
# #     # Use pytesseract to extract text from the image
# #     extracted_text = pytesseract.image_to_string(img)
    
# #     # List to store processed results
# #     combined_text_list = []

# #     # Regular expression to match unwanted numeric patterns
# #     unwanted_values = [
# #         "805 00 0 00",
# #         "805 00",
# #         "199 00 0 00",
# #         "796 00",
# #         "16000 00 0 00 1600 00",
# #         "0 00",
# #     ]
    
# #     # Split the text into lines and process each line
# #     for line in extracted_text.split('\n'):
# #         # Clean the text by removing unwanted symbols
# #         cleaned_line = re.sub(r'[$.,\.]', '', line).strip() 
        
# #         # Skip lines with unnecessary multiple commas or empty values
# #         if re.search(r',\s*,', cleaned_line):
# #             continue
        
# #         # Skip lines that match unwanted patterns
# #         if any(value in cleaned_line for value in unwanted_values):
# #             continue
        
# #         # Extract numbers from the line
# #         extracted_numbers = ' '.join(re.findall(r'\d+', line))  # Extract numbers
# #         cleaned_text = re.sub(r'\d+', '', cleaned_line).strip()  # Remove numbers from text
        
# #         # Combine text and numbers in the format you requested
# #         if extracted_numbers:
# #             combined_text_list.append(f'{cleaned_text}, {extracted_numbers}')
# #         else:
# #             if cleaned_text:  # Add only non-empty lines without numbers
# #                 combined_text_list.append(cleaned_text)
    
# #     return combined_text_list


# # @app.post("/extract-text/")
# # async def extract_text_from_image(file: UploadFile = File(...)):
# #     # Save the uploaded image
# #     image_path = os.path.join(UPLOAD_DIR, file.filename)
# #     with open(image_path, "wb") as f:
# #         f.write(await file.read())
    
# #     # Extract text from the image
# #     combined_text = extract_text(image_path)
    
# #     # Convert the results into a DataFrame for Excel
# #     data = {'combined_text': combined_text}
# #     df = pd.DataFrame(data)
    
# #     # Save the DataFrame to an Excel file
# #     df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')

# #     # Create a JSON response with a download link to the file
# #     response = {
# #         "extracted_data": combined_text,
# #     }

# #     return JSONResponse(content=response)

# # @app.get("/download-excel")
# # async def download_excel():
# #     # Return the Excel file as a downloadable response
# #     return FileResponse(EXCEL_FILE_PATH, filename="Final_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import cv2
# import pytesseract
# import re
# import pandas as pd
# import os

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Path to save the uploaded image
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Path to save the generated Excel file
# EXCEL_FILE_PATH = "Final_extracted_data.xlsx"

# def preprocess_image(image_path):
#     # Load the image in color
#     img = cv2.imread(image_path)
    
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Otsu's binary thresholding
#     _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Apply adaptive thresholding for local contrast enhancement
#     adaptive_thresh = cv2.adaptiveThreshold(binary_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                             cv2.THRESH_BINARY, 11, 2)
    
#     # Use median blurring to reduce noise
#     blurred_img = cv2.medianBlur(adaptive_thresh, 3)
    
#     # Apply morphological transformation to enhance text contours
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     morph_img = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, kernel)
    
#     return morph_img

# def extract_text(image_path):
#     # Preprocess the image
#     processed_img = preprocess_image(image_path)
    
#     # Use pytesseract to extract text from the processed image
#     extracted_text = pytesseract.image_to_string(processed_img)
    
#     # List to store processed results
#     combined_text_list = []

#     # Regular expression to match unwanted numeric patterns
#     unwanted_values = [
#         "805 00 0 00",
#         "805 00",
#         "199 00 0 00",
#         "796 00",
#         "16000 00 0 00 1600 00",
#         "0 00",
#     ]
    
#     # Split the text into lines and process each line
#     for line in extracted_text.split('\n'):
#         # Clean the text by removing unwanted symbols
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip() 
        
#         # Skip lines with unnecessary multiple commas or empty values
#         if re.search(r',\s*,', cleaned_line):
#             continue
        
#         # Skip lines that match unwanted patterns
#         if any(value in cleaned_line for value in unwanted_values):
#             continue
        
#         # Extract numbers from the line
#         extracted_numbers = ' '.join(re.findall(r'\d+', line))  # Extract numbers
#         cleaned_text = re.sub(r'\d+', '', cleaned_line).strip()  # Remove numbers from text
        
#         # Combine text and numbers in the format you requested
#         if extracted_numbers:
#             combined_text_list.append(f'{cleaned_text}, {extracted_numbers}')
#         else:
#             if cleaned_text:  # Add only non-empty lines without numbers
#                 combined_text_list.append(cleaned_text)
    
#     return combined_text_list

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     # Save the uploaded image
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract text from the image
#     combined_text = extract_text(image_path)
    
#     # Convert the results into a DataFrame for Excel
#     df = pd.DataFrame(combined_text, columns=['Extracted Data'])
    
#     # Save the DataFrame to an Excel file
#     df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')

#     # Create a JSON response with a download link to the file
#     response = {
#         "extracted_data": combined_text,
#     }

#     return JSONResponse(content=response)

# @app.get("/download-excel")
# async def download_excel():
#     # Return the Excel file as a downloadable response
#     return FileResponse(EXCEL_FILE_PATH, filename="Final_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import pytesseract
# import cv2
# import os
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# # from fastapi import FastAPI, File, UploadFile
# # from fastapi.responses import JSONResponse, FileResponse
# # import cv2
# # import pytesseract
# import re
# import pandas as pd
# import os

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Path to save the uploaded image
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Function to preprocess the image
# def preprocess_image(image_path):
#     """
#     Preprocess the image to enhance text clarity and contrast for better OCR performance.
#     """
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply Otsu's binary thresholding for better contrast
#     _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Use median blurring to reduce noise
#     processed_img = cv2.medianBlur(binary_img, 3)
    
#     return processed_img

# # Function to extract text with confidence check
# def extract_text_with_confidence(image_path, confidence_threshold=90, retries=3):
#     """
#     Extracts text from an image using OCR and ensures the average confidence is above the given threshold.
#     Retries OCR if the confidence is too low.
#     """
#     for _ in range(retries):
#         # Preprocess the image for better OCR results
#         preprocessed_image = preprocess_image(image_path)
        
#         # Perform OCR with data output, which includes the confidence scores
#         ocr_result = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.DICT)
        
#         # Extract the confidence values, filtering out invalid ones ('-1')
#         confidences = [int(conf) for conf in ocr_result['conf'] if conf != '-1']
        
#         if confidences:
#             avg_confidence = sum(confidences) / len(confidences)
#         else:
#             avg_confidence = 0
        
#         print(f"Average OCR Confidence: {avg_confidence}%")
        
#         # If confidence meets threshold, return extracted text
#         if avg_confidence >= confidence_threshold:
#             print("Confidence is above the threshold. Proceeding with text extraction.")
#             extracted_text = " ".join([ocr_result['text'][i] for i in range(len(ocr_result['text'])) if int(ocr_result['conf'][i]) != '-1'])
#             return extracted_text
        
#         print("Confidence is below threshold, retrying...")
    
#     # If maximum retries reached or confidence doesn't meet threshold, return best available text
#     print("Maximum retries reached. Returning best attempt.")
#     extracted_text = " ".join([ocr_result['text'][i] for i in range(len(ocr_result['text'])) if int(ocr_result['conf'][i]) != '-1'])
#     return extracted_text

# # FastAPI route to handle image uploads and text extraction
# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     # Save the uploaded image
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract text from the image with confidence check
#     extracted_text = extract_text_with_confidence(image_path)
    
#     # Return the extracted text in the response
#     response = {
#         "extracted_data": extracted_text,
#     }

#     return JSONResponse(content=response)

# Run the FastAPI application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# import cv2
# import numpy as np
# import pytesseract
# import os
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# import re
# from textblob import TextBlob  # Importing TextBlob for spell checking

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Path to save the uploaded image and Excel file
# UPLOAD_DIR = "uploads/"
# EXCEL_FILE_PATH = "Final_extracted_data.xlsx"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     """Apply brightness and contrast to improve image visibility."""
#     return cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)

# def gamma_correction(img, gamma=1.5):
#     """Apply gamma correction to adjust brightness and contrast."""
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def preprocess_image(image_path):
#     """Preprocess the image to improve text extraction accuracy."""
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     avg_brightness = np.mean(gray)
#     if avg_brightness < 100:  # Dark image handling
#         img = gamma_correction(adjust_brightness_contrast(gray))
#     else:  # Bright enough image
#         img = gray

#     _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return binary_img

# def correct_spelling(text):
#     """Correct spelling using TextBlob."""
#     return str(TextBlob(text).correct())

# def clean_text(extracted_text):
#     """Clean and preprocess the extracted text."""
#     unwanted_values = [
#         "805 00 0 00", "805 00", "199 00 0 00", "796 00",
#         "16000 00 0 00 1600 00", "0 00"
#     ]
    
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
        
#         cleaned_line = correct_spelling(cleaned_line)  # Spelling correction
#         numbers = ' '.join(re.findall(r'\d+', line))  # Extract numbers
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()  # Remove numbers from text
        
#         if cleaned_line:  # Append only non-empty cleaned lines
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)

#     return processed_lines

# def extract_text(image_path):
#     """Extract text from image after preprocessing and cleaning."""
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
#     return clean_text(extracted_text)

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     """Extract text from uploaded image and save to an Excel file."""
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract text and prepare the DataFrame
#     combined_text = extract_text(image_path)
#     df = pd.DataFrame({'combined_text': combined_text})

#     # Save DataFrame to Excel
#     df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')

#     # Return JSON response with extracted data
#     return JSONResponse(content={"extracted_data": combined_text})

# @app.get("/download-excel")
# async def download_excel():
#     """Allow the user to download the generated Excel file."""
#     return FileResponse(EXCEL_FILE_PATH, filename="Final_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob  # Importing TextBlob for spell checking

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Path to save the uploaded image
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Path to save the generated Excel file
# EXCEL_FILE_PATH = "Final_extracted_data.xlsx"

# # Example custom list of correct words (expand as needed)
# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     # Apply brightness and contrast to improve visibility
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     # Apply gamma correction to adjust brightness and contrast
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def preprocess_image(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Calculate the average brightness of the image
#     avg_brightness = np.mean(gray)

#     # If the image is dark (below a threshold), apply enhancements
#     if avg_brightness < 100:  # Threshold can be adjusted
#         print("Image is dark, applying brightness and contrast adjustments.")
        
#         # Apply brightness and contrast adjustment
#         img_adjusted = adjust_brightness_contrast(gray)
        
#         # Apply gamma correction to improve the image
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=1.5)
        
#         # Apply Otsu's thresholding to make it binary (black and white)
#         _, binary_img = cv2.threshold(img_gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         print("Image brightness is sufficient, no adjustments needed.")
        
#         # If the image is bright enough, just use Otsu's thresholding
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = []
#     for word in words:
#         # Find the closest match to the word in the list of correct words
#         closest_match = process.extractOne(word, correct_words)
        
#         # If a match is found and the match score is sufficiently high, use the corrected word
#         if closest_match and closest_match[1] >= 80:  # Adjust score threshold as needed
#             corrected_words.append(closest_match[0])
#         else:
#             corrected_words.append(word)  # If no match, keep the word as is
#     return " ".join(corrected_words)

# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)  # Spelling correction with context
#         numbers = ' '.join(re.findall(r'\d+', line))  
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()  
#         if cleaned_line:  
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
#     return clean_text(extracted_text)

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     # Save the uploaded image
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract text from the image
#     combined_text = extract_text(image_path)
    
#     # Convert the results into a DataFrame for Excel
#     data = {'combined_text': combined_text}
#     df = pd.DataFrame(data)
    
#     # Save the DataFrame to an Excel file
#     df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')

#     # Create a JSON response with the extracted data
#     response = {
#         "extracted_data": combined_text,
#     }

#     return JSONResponse(content=response)

# @app.get("/download-excel")
# async def download_excel():
#     # Return the Excel file as a downloadable response
#     return FileResponse(EXCEL_FILE_PATH, filename="Final_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob  # Importing TextBlob for spell checking

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Path to save the uploaded image
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Path to save the generated Excel file
# EXCEL_FILE_PATH = "Final_extracted_data.xlsx"

# # Example custom list of correct words (expand as needed)
# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]

# # Default values for brightness, gamma, and saturation
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     # Apply brightness and contrast to improve visibility
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     # Apply gamma correction to adjust brightness and contrast
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     # Convert image to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  # Adjust the saturation channel
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # Ensure saturation stays within bounds
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Calculate the average brightness of the image
#     avg_brightness = np.mean(gray)

#     # If the image is dark (below a threshold), apply enhancements
#     if avg_brightness < 100:  # Threshold can be adjusted
#         print("Image is dark, applying brightness and contrast adjustments.")
        
#         # Apply brightness and contrast adjustment
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
        
#         # Apply gamma correction to improve the image
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
        
#         # Apply saturation adjustment
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
        
#         # Apply Otsu's thresholding to make it binary (black and white)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         print("Image brightness is sufficient, no adjustments needed.")
        
#         # If the image is bright enough, just use Otsu's thresholding
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = []
#     for word in words:
#         # Use TextBlob to correct the spelling of the word
#         corrected_word = TextBlob(word).correct()
#         corrected_words.append(str(corrected_word))
#     return " ".join(corrected_words)

# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)  # Spelling correction with context
#         numbers = ' '.join(re.findall(r'\d+', line))  
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()  
#         if cleaned_line:  
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# def fuzzy_match_keywords(text, correct_words):
#     # Perform fuzzy matching to find the closest matches for keywords
#     matched_keywords = []
#     words_in_text = text.split()
    
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:  # Consider a match only if similarity score is greater than 80
#             matched_keywords.append(best_match[0])
    
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
    
#     # Perform fuzzy matching to find important keywords
#     matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
    
#     return clean_text(extracted_text), matched_keywords

# @app.post("/extract-text/")  # Extract text from the uploaded image
# async def extract_text_from_image(file: UploadFile = File(...)):
#     # Save the uploaded image
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract text from the image
#     cleaned_text, matched_keywords = extract_text(image_path)
    
#     # Convert the results into a DataFrame for Excel
#     data = {
#         'cleaned_text': cleaned_text,
#         'matched_keywords': matched_keywords
#     }
#     df = pd.DataFrame(data)
    
#     # Save the DataFrame to an Excel file
#     df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')

#     # Create a JSON response with the extracted data
#     response = {
#         "extracted_data": cleaned_text,
#         "matched_keywords": matched_keywords
#     }

#     return JSONResponse(content=response)

# @app.get("/download-excel")  # Download the generated Excel file
# async def download_excel():
#     # Return the Excel file as a downloadable response
#     return FileResponse(EXCEL_FILE_PATH, filename="Final_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob  # Importing TextBlob for spell checking

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Path to save the uploaded image
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Path to save the generated Excel file
# EXCEL_FILE_PATH = "Final_extracted_data.xlsx"

# # Example custom list of correct words (expand as needed)
# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]

# # Default values for brightness, gamma, and saturation
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     # Apply brightness and contrast to improve visibility
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     # Apply gamma correction to adjust brightness and contrast
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     # Convert image to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  # Adjust the saturation channel
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # Ensure saturation stays within bounds
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Calculate the average brightness of the image
#     avg_brightness = np.mean(gray)

#     # If the image is dark (below a threshold), apply enhancements
#     if avg_brightness < 100:  # Threshold can be adjusted
#         print("Image is dark, applying brightness and contrast adjustments.")
        
#         # Apply brightness and contrast adjustment
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
        
#         # Apply gamma correction to improve the image
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
        
#         # Apply saturation adjustment
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
        
#         # Apply Otsu's thresholding to make it binary (black and white)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         print("Image brightness is sufficient, no adjustments needed.")
        
#         # If the image is bright enough, just use Otsu's thresholding
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = []
#     for word in words:
#         # Use TextBlob to correct the spelling of the word
#         corrected_word = TextBlob(word).correct()
#         corrected_words.append(str(corrected_word))
#     return " ".join(corrected_words)

# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)  # Spelling correction with context
#         numbers = ' '.join(re.findall(r'\d+', line))  
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()  
#         if cleaned_line:  
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# def fuzzy_match_keywords(text, correct_words):
#     # Perform fuzzy matching to find the closest matches for keywords
#     matched_keywords = []
#     words_in_text = text.split()
    
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:  # Consider a match only if similarity score is greater than 80
#             matched_keywords.append(best_match[0])
    
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
    
#     # Perform fuzzy matching to find important keywords
#     matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
    
#     return clean_text(extracted_text), matched_keywords

# @app.post("/extract-text/")  # Extract text from the uploaded image
# async def extract_text_from_image(file: UploadFile = File(...)):
#     # Save the uploaded image
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract text from the image
#     cleaned_text, matched_keywords = extract_text(image_path)
    
#     # Ensure both lists have the same length by padding with None or truncating
#     max_length = max(len(cleaned_text), len(matched_keywords))
    
#     # Pad the shorter list with None
#     cleaned_text.extend([None] * (max_length - len(cleaned_text)))
    
#     # Convert the results into a DataFrame for Excel (only include cleaned_text)
#     data = {
#         'cleaned_text': cleaned_text,
#     }
    
#     df = pd.DataFrame(data)
    
#     # Save the DataFrame to an Excel file
#     df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')

#     # Create a JSON response with the extracted text only
#     response = {
#         "extracted_data": cleaned_text  # Only send back the cleaned text in the response
#     }

#     return JSONResponse(content=response)

# @app.get("/download-excel")  # Download the generated Excel file
# async def download_excel():
#     # Return the Excel file as a downloadable response
#     return FileResponse(EXCEL_FILE_PATH, filename="Final_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob  # Importing TextBlob for spell checking

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Path to save the uploaded image
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Path to save the generated Excel file
# EXCEL_FILE_PATH = "Final_extracted_data.xlsx"

# # Example custom list of correct words (expand as needed)
# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]

# # Default values for brightness, gamma, and saturation
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     # Apply brightness and contrast to improve visibility
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     # Apply gamma correction to adjust brightness and contrast
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     # Convert image to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  # Adjust the saturation channel
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # Ensure saturation stays within bounds
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Calculate the average brightness of the image
#     avg_brightness = np.mean(gray)

#     # If the image is dark (below a threshold), apply enhancements
#     if avg_brightness < 100:  # Threshold can be adjusted
#         print("Image is dark, applying brightness and contrast adjustments.")
        
#         # Apply brightness and contrast adjustment
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
        
#         # Apply gamma correction to improve the image
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
        
#         # Apply saturation adjustment
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
        
#         # Apply Otsu's thresholding to make it binary (black and white)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         print("Image brightness is sufficient, no adjustments needed.")
        
#         # If the image is bright enough, just use Otsu's thresholding
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = []
#     for word in words:
#         # Use TextBlob to correct the spelling of the word
#         corrected_word = TextBlob(word).correct()
#         corrected_words.append(str(corrected_word))
#     return " ".join(corrected_words)

# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)  # Spelling correction with context
#         numbers = ' '.join(re.findall(r'\d+', line))  
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()  
#         if cleaned_line:  
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# def fuzzy_match_keywords(text, correct_words):
#     # Perform fuzzy matching to find the closest matches for keywords
#     matched_keywords = []
#     words_in_text = text.split()
    
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:  # Consider a match only if similarity score is greater than 80
#             matched_keywords.append(best_match[0])
    
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
    
#     # Perform fuzzy matching to find important keywords
#     matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
    
#     return clean_text(extracted_text), matched_keywords

# @app.post("/extract-text/")  # Extract text from the uploaded image
# async def extract_text_from_image(file: UploadFile = File(...)):
#     # Save the uploaded image
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract text from the image
#     cleaned_text, matched_keywords = extract_text(image_path)
    
#     # Ensure both lists have the same length by padding with None or truncating
#     max_length = max(len(cleaned_text), len(matched_keywords))
    
#     # Pad the shorter list with None
#     cleaned_text.extend([None] * (max_length - len(cleaned_text)))
    
#     # Convert the results into a DataFrame for Excel (only include cleaned_text)
#     data = {
#         'cleaned_text': cleaned_text,
#     }
    
#     df = pd.DataFrame(data)
    
#     # Save the DataFrame to an Excel file
#     df.to_excel(EXCEL_FILE_PATH, index=False, engine='openpyxl')

#     # Create a JSON response with the extracted text only
#     response = {
#         "extracted_data": cleaned_text  # Only send back the cleaned text in the response
#     }

#     return JSONResponse(content=response)

# @app.get("/download-excel")  # Download the generated Excel file
# async def download_excel():
#     # Return the Excel file as a downloadable response
#     return FileResponse(EXCEL_FILE_PATH, filename="Final_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob
# import openpyxl
# from openpyxl.worksheet.table import Table, TableStyleInfo

# # Set Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# # Initialize FastAPI app
# app = FastAPI()

# # Directory for file uploads
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Define an empty string to hold the latest Excel file path
# latest_excel_file_path = ""

# # List of correct words for fuzzy matching
# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# # Helper functions
# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = [str(TextBlob(word).correct()) for word in words]
#     return " ".join(corrected_words)

# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)
#         numbers = ' '.join(re.findall(r'\d+', line))
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()
#         if cleaned_line:
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# def fuzzy_match_keywords(text, correct_words):
#     matched_keywords = []
#     words_in_text = text.split()
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:
#             matched_keywords.append(best_match[0])
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
#     matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
#     return clean_text(extracted_text), matched_keywords

# # API endpoints
# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     global latest_excel_file_path  # Use global to track the latest file path
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     cleaned_text, matched_keywords = extract_text(image_path)
    
#     text_only = []
#     numbers_only = []
#     for entry in cleaned_text:
#         if entry:
#             text_part = re.sub(r'[\d\W_]+$', '', entry).strip()
#             number_part = ' '.join(re.findall(r'\b\d+[.,]?\d*\b', entry))
#             text_only.append(text_part)
#             numbers_only.append(number_part)
    
#     data = {'Description': text_only, 'Amount': numbers_only}
#     df = pd.DataFrame(data)
    
#     # Save DataFrame to an Excel file
#     latest_excel_file_path = os.path.join(UPLOAD_DIR, "Text_extracted_data.xlsx")
#     df.to_excel(latest_excel_file_path, index=False, engine='openpyxl')
    
#     # Load the saved file to apply table formatting
#     wb = openpyxl.load_workbook(latest_excel_file_path)
#     ws = wb.active

#     # Define the range of the table (based on data size)
#     table_range = f"A1:B{len(df) + 1}"

#     # Create a table
#     table = Table(displayName="ExtractedData", ref=table_range)

#     # Add a style to the table
#     style = TableStyleInfo(
#         name="TableStyleMedium9",  # Choose a predefined style
#         showFirstColumn=False,
#         showLastColumn=False,
#         showRowStripes=True,
#         showColumnStripes=True,
#     )
#     table.tableStyleInfo = style

#     # Add the table to the worksheet
#     ws.add_table(table)

#     # Save the workbook with the formatted table
#     wb.save(latest_excel_file_path)

#     return JSONResponse(content={
#         "extracted_data": cleaned_text,
#         "excel_file": latest_excel_file_path
#     })

# @app.get("/download-excel")
# async def download_excel():
#     # Download the latest Excel file saved
#     if not latest_excel_file_path:
#         return JSONResponse(content={"error": "No file found to download."}, status_code=404)
#     return FileResponse(latest_excel_file_path, filename="Text_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")












# import cv2
# import pytesseract
# import os
# import numpy as np
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob
# import re
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]

# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# # Rescaling function for better recognition
# def resize_image(image, scale_factor=2):
#     width = int(image.shape[1] * scale_factor)
#     height = int(image.shape[0] * scale_factor)
#     dim = (width, height)
#     return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

# # Adaptive thresholding for better text-background distinction
# def adaptive_thresholding(image):
#     return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY, 11, 2)

# # Denoising function to reduce noise in the image
# def denoise_image(image):
#     return cv2.medianBlur(image, 5)

# # Deskewing function to correct image orientation
# def deskew_image(image):
#     coords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
#                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated_image

# # Preprocessing pipeline with all improvements
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img_resized = resize_image(img)
#     img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
#     img_denoised = denoise_image(img_gray)
#     img_deskewed = deskew_image(img_denoised)
#     img_thresholded = adaptive_thresholding(img_deskewed)
    
#     return img_thresholded

# # Spell correction with TextBlob
# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = [str(TextBlob(word).correct()) for word in words]
#     return " ".join(corrected_words)

# # Clean extracted text
# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)
#         numbers = ' '.join(re.findall(r'\d+', line))
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()
#         if cleaned_line:
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# # Fuzzy matching for keywords
# def fuzzy_match_keywords(text, correct_words):
#     matched_keywords = []
#     words_in_text = text.split()
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:
#             matched_keywords.append(best_match[0])
#     return matched_keywords

# # Extract text from image using Tesseract OCR with configuration options
# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     custom_config = r'--oem 1 --psm 6'  # Using LSTM-based OCR engine and page segmentation mode
#     extracted_text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
#     matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
#     return clean_text(extracted_text), matched_keywords

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     global latest_excel_file_path  # Use global to track the latest file path
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     cleaned_text, matched_keywords = extract_text(image_path)
    
#     text_only = []
#     numbers_only = []
#     for entry in cleaned_text:
#         if entry:
#             text_part = re.sub(r'[\d\W_]+$', '', entry).strip()
#             number_part = ' '.join(re.findall(r'\b\d+[.,]?\d*\b', entry))
#             text_only.append(text_part)
#             numbers_only.append(number_part)
    
#     data = {'Description': text_only, 'Amount': numbers_only}
#     df = pd.DataFrame(data)
    
#     latest_excel_file_path = os.path.join(UPLOAD_DIR, "Text_extracted_data.xlsx")
#     df.to_excel(latest_excel_file_path, index=False, engine='openpyxl')
    
#     return JSONResponse(content={
#         "extracted_data": cleaned_text,
#         "excel_file": latest_excel_file_path
#     })

# @app.get("/download-excel")
# async def download_excel():
#     # Download the latest Excel file saved
#     if not latest_excel_file_path:
#         return JSONResponse(content={"error": "No file found to download."}, status_code=404)
#     return FileResponse(latest_excel_file_path, filename="Text_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob  # Importing TextBlob for spell checking

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Path to save the uploaded image
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Path to save the generated Excel file
# EXCEL_FILE_PATH = "Final_extracted_data.xlsx"

# # Example custom list of correct words (expand as needed)
# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]

# # Default values for brightness, gamma, and saturation
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     # Apply brightness and contrast to improve visibility
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     # Apply gamma correction to adjust brightness and contrast
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     # Convert image to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  # Adjust the saturation channel
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # Ensure saturation stays within bounds
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Calculate the average brightness of the image
#     avg_brightness = np.mean(gray)

#     # If the image is dark (below a threshold), apply enhancements
#     if avg_brightness < 100:  # Threshold can be adjusted
#         print("Image is dark, applying brightness and contrast adjustments.")
        
#         # Apply brightness and contrast adjustment
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
        
#         # Apply gamma correction to improve the image
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
        
#         # Apply saturation adjustment
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
        
#         # Apply Otsu's thresholding to make it binary (black and white)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         print("Image brightness is sufficient, no adjustments needed.")
        
#         # If the image is bright enough, just use Otsu's thresholding
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = []
#     for word in words:
#         # Use TextBlob to correct the spelling of the word
#         corrected_word = TextBlob(word).correct()
#         corrected_words.append(str(corrected_word))
#     return " ".join(corrected_words)

# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)  # Spelling correction with context
#         numbers = ' '.join(re.findall(r'\d+', line))  
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()  
#         if cleaned_line:  
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# def fuzzy_match_keywords(text, correct_words):
#     # Perform fuzzy matching to find the closest matches for keywords
#     matched_keywords = []
#     words_in_text = text.split()
    
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:  # Consider a match only if similarity score is greater than 80
#             matched_keywords.append(best_match[0])
    
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
    
#     # Perform fuzzy matching to find important keywords
#     matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
    
#     return clean_text(extracted_text), matched_keywords

# @app.post("/extract-text/")  # Extract text from the uploaded image
# async def extract_text_from_image(file: UploadFile = File(...)):
#     # Save the uploaded image
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract text from the image
#     cleaned_text, matched_keywords = extract_text(image_path)
    
#     # Separate potential text and number fields in each entry
#     text_only = []
#     numbers_only = []
    
#     for entry in cleaned_text:
#         if entry:
#             # Extract textual part and numerical part using regex
#             text_part = re.sub(r'[\d\W_]+$', '', entry).strip()  # Remove trailing numbers or symbols for text
#             number_part = ' '.join(re.findall(r'\b\d+[.,]?\d*\b', entry))  # Match numbers (integers or decimals)
            
#             text_only.append(text_part)
#             numbers_only.append(number_part)
    
#     # Convert the results into a DataFrame for Excel
#     data = {'Description': text_only, 'Amount': numbers_only}
#     df = pd.DataFrame(data)
    
#     # Save the Excel file
#     excel_path = os.path.join(UPLOAD_DIR, "Final_extracted_data.xlsx")
#     df.to_excel(excel_path, index=False, engine='openpyxl')
    
#     # Return the path to the saved Excel file
#     return JSONResponse(content={
#         "extracted_data": cleaned_text,
#         "excel_file": excel_path
#     })

# @app.get("/download-excel")  # Download the generated Excel file
# async def download_excel():
#     # Return the Excel file as a downloadable response
#     return FileResponse(EXCEL_FILE_PATH, filename="Final_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Directory and file paths
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# EXCEL_FILE_PATH = os.path.join(UPLOAD_DIR, "Final_extracted_data.xlsx")

# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def deskew_image(img):
#     # Deskew the image to improve OCR accuracy
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     coords = np.column_stack(np.where(gray > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = deskew_image(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img = gamma_correction(img, gamma=GAMMA)
#         img = adjust_saturation(img, saturation=SATURATION)
#         _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return img

# def extract_invoice_table(text):
#     # Extract tabular data for invoice-like documents
#     lines = text.split('\n')
#     table_data = []
#     for line in lines:
#         # Regex to extract line items with description, quantity, price, and total
#         match = re.match(r"(.*?)(\d+\s*\d*)(\d+\.\d+)", line)
#         if match:
#             description = match.group(1).strip()
#             quantity = match.group(2).strip()
#             amount = match.group(3).strip()
#             table_data.append({"Description": description, "Quantity": quantity, "Amount": amount})
#     return table_data

# def extract_payment_form_details(text):
#     # Extract fields like account number, payment date, and amount from payment forms
#     details = {}
#     account_match = re.search(r"Account Number:\s*(\d+)", text, re.IGNORECASE)
#     date_match = re.search(r"Date:\s*([\d/]+)", text, re.IGNORECASE)
#     amount_match = re.search(r"Amount:\s*\$?([\d,\.]+)", text, re.IGNORECASE)

#     if account_match:
#         details["Account Number"] = account_match.group(1)
#     if date_match:
#         details["Payment Date"] = date_match.group(1)
#     if amount_match:
#         details["Payment Amount"] = amount_match.group(1)

#     return details

# @app.post("/extract-invoice/")
# async def extract_invoice_data(file: UploadFile = File(...)):
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())

#     img = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(img)
#     table_data = extract_invoice_table(extracted_text)

#     df = pd.DataFrame(table_data)
#     df.to_excel(EXCEL_FILE_PATH, index=False, engine="openpyxl")

#     return JSONResponse({
#         "message": "Invoice data extracted successfully.",
#         "excel_path": EXCEL_FILE_PATH
#     })

# @app.post("/extract-payment-form/")
# async def extract_payment_form(file: UploadFile = File(...)):
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())

#     img = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(img)
#     payment_details = extract_payment_form_details(extracted_text)

#     return JSONResponse({
#         "message": "Payment form details extracted successfully.",
#         "payment_details": payment_details
#     })

# @app.get("/download-excel/")
# async def download_excel():
#     return FileResponse(EXCEL_FILE_PATH, filename="Extracted_Data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# Import necessary libraries
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob  # Importing TextBlob for spell checking

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Paths and directories
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# EXCEL_FILE_PATH = os.path.join(UPLOAD_DIR, "Final_extracted_data.xlsx")

# # Default processing values
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# # Preprocessing functions
# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast / 127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)
    
#     if avg_brightness < 100:
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
#         _, binary_img = cv2.threshold(img_gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# # Dynamic table extraction function
# def extract_table_with_dynamic_headers(image_path):
#     """
#     Extracts table data from the image using the headers dynamically detected from the invoice photo.
#     """
#     # Preprocess image
#     preprocessed_img = preprocess_image(image_path)
    
#     # Extract text from the preprocessed image
#     extracted_text = pytesseract.image_to_string(preprocessed_img, config="--psm 6")
#     lines = extracted_text.split("\n")
    
#     headers = []
#     table_data = []
    
#     for line in lines:
#         # Normalize whitespace and clean the line
#         line = re.sub(r"\s{2,}", "\t", line.strip())  # Replace multiple spaces with tabs
        
#         # Check for headers (based on patterns or position)
#         if not headers and re.search(r'NO\.|ITEM|RATE|QTY\.|PRICE', line, re.IGNORECASE):
#             headers = line.split("\t")
#             headers = [header.strip() for header in headers]  # Clean up headers
#             continue
        
#         # Process rows if headers are identified
#         if headers:
#             row = line.split("\t")
#             if len(row) == len(headers):  # Ensure row aligns with headers
#                 table_data.append([col.strip() for col in row])
#         else:
#             # Fallback: Collect rows if no headers are detected yet
#             row = line.split("\t")
#             table_data.append([col.strip() for col in row])
    
#     # Create DataFrame
#     if headers:
#         df = pd.DataFrame(table_data, columns=headers)
#     else:
#         df = pd.DataFrame(table_data)  # Create DataFrame without column names if no headers found
    
#     return df

# # Dynamic invoice extraction endpoint
# @app.post("/extract-invoice-dynamic/")
# async def extract_invoice_dynamic(file: UploadFile = File(...)):
#     """
#     Extracts table data from an invoice image, dynamically using the headers in the photo.
#     """
#     # Save the uploaded image
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract table data dynamically
#     df = extract_table_with_dynamic_headers(image_path)
    
#     # Save to Excel
#     df.to_excel(EXCEL_FILE_PATH, index=False, engine="openpyxl")
    
#     return JSONResponse({
#         "message": "Invoice data extracted successfully.",
#         "headers": list(df.columns) if not df.empty else "No headers detected",
#         "excel_path": EXCEL_FILE_PATH
#     })

# # Download Excel endpoint
# @app.get("/download-excel/")
# async def download_excel():
#     return FileResponse(EXCEL_FILE_PATH, filename="Extracted_Data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from datetime import datetime

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Directory and file paths
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img = gamma_correction(img, gamma=GAMMA)
#         img = adjust_saturation(img, saturation=SATURATION)
#         _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return img

# def extract_table_with_dynamic_headers(image_path):
#     preprocessed_img = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_img, config="--psm 6")
#     lines = extracted_text.split("\n")
    
#     headers = []
#     table_data = []
    
#     for line in lines:
#         line = re.sub(r"\s{2,}", "\t", line.strip())
        
#         # Check if the line contains potential headers (keywords such as ITEM, RATE, QTY, etc.)
#         if not headers and re.search(r'NO\.|ITEM|RATE|QTY\.|PRICE', line, re.IGNORECASE):
#             headers = line.split("\t")
#             headers = [header.strip() for header in headers]
#             continue
        
#         if headers:
#             row = line.split("\t")
#             if len(row) == len(headers):
#                 table_data.append([col.strip() for col in row])
#         else:
#             row = line.split("\t")
#             table_data.append([col.strip() for col in row])
    
#     if headers:
#         df = pd.DataFrame(table_data, columns=headers)
#     else:
#         df = pd.DataFrame(table_data)
    
#     return df, table_data, headers  # Return headers and raw table data as well

# @app.post("/extract-invoice-dynamic/")
# async def extract_invoice_dynamic(file: UploadFile = File(...)):
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     image_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
#     with open(image_path, "wb") as f:
#         f.write(await file.read())

#     df, table_data, headers = extract_table_with_dynamic_headers(image_path)

#     excel_file_path = os.path.join(UPLOAD_DIR, f"Invoice_Data_{timestamp}.xlsx")
#     df.to_excel(excel_file_path, index=False, engine="openpyxl")

#     # Format extracted data as a list of dictionaries for the response
#     extracted_data = [{"Row": idx + 1, **dict(zip(headers, row))} for idx, row in enumerate(table_data)]

#     return JSONResponse({
#         "message": "Invoice data extracted successfully.",
#         "headers": headers if headers else "No headers detected",
#         "extracted_data": extracted_data,  # Include the extracted table data
#         "excel_path": excel_file_path
#     })

# @app.get("/download-excel/")
# async def download_excel(file_name: str):
#     file_path = os.path.join(UPLOAD_DIR, file_name)
#     if os.path.exists(file_path):
#         return FileResponse(file_path, filename=file_name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
#     return JSONResponse({"error": "File not found"}, status_code=404)
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from datetime import datetime

# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# # Directory and file paths
# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img = gamma_correction(img, gamma=GAMMA)
#         img = adjust_saturation(img, saturation=SATURATION)
#         _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return img

# def extract_table_with_dynamic_headers(image_path):
#     preprocessed_img = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_img, config="--psm 6")
#     lines = extracted_text.split("\n")
    
#     headers = []
#     table_data = []
    
#     for line in lines:
#         line = re.sub(r"\s{2,}", "\t", line.strip())
        
#         # Check if the line contains potential headers (keywords such as ITEM, RATE, QTY, etc.)
#         if not headers and re.search(r'NO\.|ITEM|RATE|QTY\.|PRICE', line, re.IGNORECASE):
#             headers = line.split("\t")
#             headers = [header.strip() for header in headers]
#             continue
        
#         if headers:
#             row = line.split("\t")
#             if len(row) == len(headers):
#                 table_data.append([col.strip() for col in row])
#         else:
#             row = line.split("\t")
#             table_data.append([col.strip() for col in row])
    
#     if headers:
#         df = pd.DataFrame(table_data, columns=headers)
#     else:
#         df = pd.DataFrame(table_data)
    
#     return df, table_data, headers  # Return headers and raw table data as well

# @app.post("/extract-invoice-dynamic/")
# async def extract_invoice_dynamic(file: UploadFile = File(...)):
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     image_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
#     with open(image_path, "wb") as f:
#         f.write(await file.read())

#     df, table_data, headers = extract_table_with_dynamic_headers(image_path)

#     excel_file_path = os.path.join(UPLOAD_DIR, f"Invoice_Data_{timestamp}.xlsx")
#     df.to_excel(excel_file_path, index=False, engine="openpyxl")

#     # Format extracted data as a list of dictionaries for the response
#     extracted_data = [{"Row": idx + 1, **dict(zip(headers, row))} for idx, row in enumerate(table_data)]

#     return JSONResponse({
#         "message": "Invoice data extracted successfully.",
#         "headers": headers if headers else "No headers detected",
#         "extracted_data": extracted_data,  # Include the extracted table data
#         "excel_path": excel_file_path
#     })

# @app.get("/download-excel/{filename}")
# async def download_excel(filename: str):
#     excel_file_path = os.path.join(UPLOAD_DIR, filename)
    
#     if os.path.exists(excel_file_path):
#         return FileResponse(excel_file_path, filename=filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
#     return JSONResponse({"error": "File not found"}, status_code=404)

# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob


# app = FastAPI()

# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Define an empty string to hold the latest Excel file path
# latest_excel_file_path = ""

# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = [str(TextBlob(word).correct()) for word in words]
#     return " ".join(corrected_words)

# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)
#         numbers = ' '.join(re.findall(r'\d+', line))
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()
#         if cleaned_line:
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# def fuzzy_match_keywords(text, correct_words):
#     matched_keywords = []
#     words_in_text = text.split()
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:
#             matched_keywords.append(best_match[0])
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
#     matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
#     return clean_text(extracted_text), matched_keywords

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     global latest_excel_file_path  # Use global to track the latest file path
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     cleaned_text, matched_keywords = extract_text(image_path)
    
#     text_only = []
#     numbers_only = []
#     for entry in cleaned_text:
#         if entry:
#             text_part = re.sub(r'[\d\W_]+$', '', entry).strip()
#             number_part = ' '.join(re.findall(r'\b\d+[.,]?\d*\b', entry))
#             text_only.append(text_part)
#             numbers_only.append(number_part)
    
#     data = {'Description': text_only, 'Amount': numbers_only}
#     df = pd.DataFrame(data)
    
#     latest_excel_file_path = os.path.join(UPLOAD_DIR, "Text_extracted_data.xlsx")
#     df.to_excel(latest_excel_file_path, index=False, engine='openpyxl')
    
#     return JSONResponse(content={
#         "extracted_data": cleaned_text,
#         "excel_file": latest_excel_file_path
#     })

# @app.get("/download-excel")
# async def download_excel():
#     # Download the latest Excel file saved
#     if not latest_excel_file_path:
#         return JSONResponse(content={"error": "No file found to download."}, status_code=404)
#     return FileResponse(latest_excel_file_path, filename="Text_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob

# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# app = FastAPI()

# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Define an empty string to hold the latest Excel file path
# latest_excel_file_path = ""

# # Keywords to look for in the invoices and payment forms
# correct_words = [
#     "Invoice", "Street", "Quantity", "Amount", "Total", "Discount", 
#     "Security", "Automation", "Solar", "Panel", "Invoice Number", 
#     "Date", "Transaction", "Amount Paid", "Payment Date", "Transaction ID"
# ]

# # Preprocessing parameters
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = [str(TextBlob(word).correct()) for word in words]
#     return " ".join(corrected_words)

# def clean_text_for_invoice(extracted_text):
#     # Invoice fields to look for (could be extended)
#     invoice_fields = ["Invoice Number", "Date", "Amount", "Total", "Quantity", "Discount"]
#     payment_fields = ["Transaction ID", "Amount Paid", "Payment Date"]
    
#     lines = extracted_text.split('\n')
#     invoice_data = {}
#     payment_data = {}

#     for line in lines:
#         line = line.strip()

#         # Check for invoice related fields
#         for field in invoice_fields:
#             if field.lower() in line.lower():
#                 invoice_data[field] = line.split(":")[-1].strip()

#         # Check for payment form related fields
#         for field in payment_fields:
#             if field.lower() in line.lower():
#                 payment_data[field] = line.split(":")[-1].strip()

#     return invoice_data, payment_data

# def fuzzy_match_keywords(text, correct_words):
#     matched_keywords = []
#     words_in_text = text.split()
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:
#             matched_keywords.append(best_match[0])
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
    
#     # Extract structured data for invoice and payment form
#     invoice_data, payment_data = clean_text_for_invoice(extracted_text)
    
#     return extracted_text, invoice_data, payment_data

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     global latest_excel_file_path  # Use global to track the latest file path
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
    
#     # Save uploaded file to disk
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract data from the image
#     extracted_text, invoice_data, payment_data = extract_text(image_path)

#     # Convert extracted data into DataFrame
#     invoice_df = pd.DataFrame([invoice_data])
#     payment_df = pd.DataFrame([payment_data])
    
#     # Save both dataframes into an Excel file
#     latest_excel_file_path = os.path.join(UPLOAD_DIR, "Extracted_Data.xlsx")
    
#     with pd.ExcelWriter(latest_excel_file_path, engine='openpyxl') as writer:
#         invoice_df.to_excel(writer, sheet_name='Invoice', index=False)
#         payment_df.to_excel(writer, sheet_name='Payment', index=False)

#     return JSONResponse(content={
#         "extracted_data": extracted_text,
#         "invoice_data": invoice_data,
#         "payment_data": payment_data,
#         "excel_file": latest_excel_file_path
#     })

# @app.get("/download-excel")
# async def download_excel():
#     # Download the latest Excel file saved
#     if not latest_excel_file_path:
#         return JSONResponse(content={"error": "No file found to download."}, status_code=404)
#     return FileResponse(latest_excel_file_path, filename="Extracted_Data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob


# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Define an empty string to hold the latest Excel file path
# latest_excel_file_path = ""

# # Keywords to look for in the invoices and payment forms
# correct_words = [
#     "Invoice", "Street", "Quantity", "Amount", "Total", "Discount", 
#     "Security", "Automation", "Solar", "Panel", "Invoice Number", 
#     "Date", "Transaction", "Amount Paid", "Payment Date", "Transaction ID"
# ]

# # Preprocessing parameters
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = [str(TextBlob(word).correct()) for word in words]
#     return " ".join(corrected_words)

# def clean_text_for_invoice(extracted_text):
#     # Define fields to extract
#     invoice_fields = ["Invoice Number", "Date", "Amount", "Total", "Quantity", "Discount"]
#     payment_fields = ["Transaction ID", "Amount Paid", "Payment Date"]
    
#     lines = extracted_text.split('\n')
#     invoice_data = {}
#     payment_data = {}

#     # Regex patterns to capture numbers, amounts, and dates
#     amount_pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"
#     date_pattern = r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}"

#     for line in lines:
#         line = line.strip()

#         # Check for invoice-related fields (e.g., "Invoice Number", "Amount")
#         for field in invoice_fields:
#             if field.lower() in line.lower():
#                 invoice_data[field] = line.split(":")[-1].strip()

#         # Capture amounts and dates using regex
#         amounts = re.findall(amount_pattern, line)
#         if amounts:
#             for amount in amounts:
#                 if "Total" in line:
#                     invoice_data["Total"] = amount
#                 elif "Amount" in line:
#                     invoice_data["Amount"] = amount

#         dates = re.findall(date_pattern, line)
#         if dates:
#             for date in dates:
#                 if "Date" in line:
#                     invoice_data["Date"] = date

#     return invoice_data, payment_data

# def fuzzy_match_keywords(text, correct_words):
#     matched_keywords = []
#     words_in_text = text.split()
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:
#             matched_keywords.append(best_match[0])
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
    
#     # Extract structured data for invoice and payment form
#     invoice_data, payment_data = clean_text_for_invoice(extracted_text)
    
#     return extracted_text, invoice_data, payment_data

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     global latest_excel_file_path  # Use global to track the latest file path
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
    
#     # Save uploaded file to disk
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract data from the image
#     extracted_text, invoice_data, payment_data = extract_text(image_path)

#     # Convert extracted data into DataFrame (structured)
#     invoice_df = pd.DataFrame([invoice_data])
#     payment_df = pd.DataFrame([payment_data])

#     # Save both dataframes into an Excel file
#     latest_excel_file_path = os.path.join(UPLOAD_DIR, "Extracted_Data.xlsx")
    
#     with pd.ExcelWriter(latest_excel_file_path, engine='openpyxl') as writer:
#         invoice_df.to_excel(writer, sheet_name='Invoice', index=False)
#         payment_df.to_excel(writer, sheet_name='Payment', index=False)

#     return JSONResponse(content={
#         "extracted_data": extracted_text,
#         "invoice_data": invoice_data,
#         "payment_data": payment_data,
#         "excel_file": latest_excel_file_path
#     })

# @app.get("/download-excel")
# async def download_excel():
#     # Download the latest Excel file saved
#     if not latest_excel_file_path:
#         return JSONResponse(content={"error": "No file found to download."}, status_code=404)
#     return FileResponse(latest_excel_file_path, filename="Extracted_Data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob


# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Define an empty string to hold the latest Excel file path
# latest_excel_file_path = ""

# # Keywords to look for in the invoices and payment forms
# correct_words = [
#     "Invoice", "Street", "Quantity", "Amount", "Total", "Discount", 
#     "Security", "Automation", "Solar", "Panel", "Invoice Number", 
#     "Date", "Transaction", "Amount Paid", "Payment Date", "Transaction ID"
# ]

# # Preprocessing parameters
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = [str(TextBlob(word).correct()) for word in words]
#     return " ".join(corrected_words)

# def clean_text_for_invoice(extracted_text):
#     # Define fields to extract
#     invoice_fields = ["Invoice Number", "Date", "Amount", "Total", "Quantity", "Discount"]
#     payment_fields = ["Transaction ID", "Amount Paid", "Payment Date"]
    
#     lines = extracted_text.split('\n')
#     invoice_data = {}
#     payment_data = {}

#     # Regex patterns to capture numbers, amounts, and dates
#     amount_pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"
#     date_pattern = r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}"

#     for line in lines:
#         line = line.strip()

#         # Check for invoice-related fields (e.g., "Invoice Number", "Amount")
#         for field in invoice_fields:
#             if field.lower() in line.lower():
#                 invoice_data[field] = line.split(":")[-1].strip()

#         # Capture amounts and dates using regex
#         amounts = re.findall(amount_pattern, line)
#         if amounts:
#             for amount in amounts:
#                 if "Total" in line:
#                     invoice_data["Total"] = amount
#                 elif "Amount" in line:
#                     invoice_data["Amount"] = amount

#         dates = re.findall(date_pattern, line)
#         if dates:
#             for date in dates:
#                 if "Date" in line:
#                     invoice_data["Date"] = date

#     return invoice_data, payment_data

# def fuzzy_match_keywords(text, correct_words):
#     matched_keywords = []
#     words_in_text = text.split()
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:
#             matched_keywords.append(best_match[0])
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
    
#     # Extract structured data for invoice and payment form
#     invoice_data, payment_data = clean_text_for_invoice(extracted_text)
    
#     return extracted_text, invoice_data, payment_data

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     global latest_excel_file_path  # Use global to track the latest file path
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
    
#     # Save uploaded file to disk
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     # Extract data from the image
#     extracted_text, invoice_data, payment_data = extract_text(image_path)

#     # Convert extracted data into DataFrame (structured)
#     invoice_df = pd.DataFrame([invoice_data])
#     payment_df = pd.DataFrame([payment_data])

#     # Save both dataframes into an Excel file
#     latest_excel_file_path = os.path.join(UPLOAD_DIR, "Extracted_Data.xlsx")
    
#     with pd.ExcelWriter(latest_excel_file_path, engine='openpyxl') as writer:
#         invoice_df.to_excel(writer, sheet_name='Invoice', index=False)
#         payment_df.to_excel(writer, sheet_name='Payment', index=False)

#     return JSONResponse(content={
#         "extracted_data": extracted_text,
#         "invoice_data": invoice_data,
#         "payment_data": payment_data,
#         "excel_file": latest_excel_file_path
#     })

# @app.get("/download-excel")
# async def download_excel():
#     # Download the latest Excel file saved
#     if not latest_excel_file_path:
#         return JSONResponse(content={"error": "No file found to download."}, status_code=404)
#     return FileResponse(latest_excel_file_path, filename="Extracted_Data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# import cv2
# import pytesseract
# import os
# import numpy as np
# import re
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse, FileResponse
# import pandas as pd
# from fuzzywuzzy import process
# from textblob import TextBlob

# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# app = FastAPI()

# UPLOAD_DIR = "uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Define an empty string to hold the latest Excel file path
# latest_excel_file_path = ""

# correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]
# BRIGHTNESS = 1.0
# GAMMA = 2.0
# SATURATION = 2.0

# def adjust_brightness_contrast(img, brightness=50, contrast=30):
#     adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
#     return adjusted

# def gamma_correction(img, gamma=1.5):
#     look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(img, look_up_table)

# def adjust_saturation(img, saturation=1.0):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = hsv[:, :, 1] * saturation  
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)

#     if avg_brightness < 100:
#         img_adjusted = adjust_brightness_contrast(gray, brightness=BRIGHTNESS, contrast=30)
#         img_gamma_corrected = gamma_correction(img_adjusted, gamma=GAMMA)
#         img_saturated = adjust_saturation(img_gamma_corrected, saturation=SATURATION)
#         _, binary_img = cv2.threshold(img_saturated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_img

# def correct_spelling_with_context(text):
#     words = text.split()
#     corrected_words = [str(TextBlob(word).correct()) for word in words]
#     return " ".join(corrected_words)

# def clean_text(extracted_text):
#     unwanted_values = ["805 00 0 00", "805 00", "199 00 0 00", "796 00", "16000 00 0 00 1600 00", "0 00"]
#     lines = extracted_text.split('\n')
#     processed_lines = []
#     for line in lines:
#         cleaned_line = re.sub(r'[$.,\.]', '', line).strip()
#         if re.search(r',\s*,', cleaned_line) or any(value in cleaned_line for value in unwanted_values):
#             continue
#         cleaned_line = correct_spelling_with_context(cleaned_line)
#         numbers = ' '.join(re.findall(r'\d+', line))
#         cleaned_line = re.sub(r'\d+', '', cleaned_line).strip()
#         if cleaned_line:
#             processed_lines.append(f'{cleaned_line}, {numbers}' if numbers else cleaned_line)
#     return processed_lines

# def fuzzy_match_keywords(text, correct_words):
#     matched_keywords = []
#     words_in_text = text.split()
#     for word in words_in_text:
#         best_match = process.extractOne(word, correct_words)
#         if best_match[1] > 80:
#             matched_keywords.append(best_match[0])
#     return matched_keywords

# def extract_text(image_path):
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(preprocessed_image)
#     matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
#     return extracted_text, matched_keywords

# def parse_table(extracted_text):
#     lines = extracted_text.split('\n')
    
#     # Look for the first row that can act as the table header
#     headers = ["Description", "Amount"]
#     for line in lines:
#         line = line.strip()
#         if "Description" in line and "Amount" in line:
#             # Found a line with both headers, use it as column headers
#             headers = [line.split()[0], line.split()[1]]  # Adjust if necessary
#             break
    
#     # Clean the text and capture the rows as data
#     rows = []
#     for line in lines:
#         if line.strip():  # Avoid empty lines
#             parts = line.split()
#             if len(parts) >= 2:  # Ensure there are at least two parts (Description and Amount)
#                 rows.append(parts)

#     return headers, rows

# @app.post("/extract-text/")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     global latest_excel_file_path  # Use global to track the latest file path
#     image_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(image_path, "wb") as f:
#         f.write(await file.read())
    
#     extracted_text, matched_keywords = extract_text(image_path)
    
#     # Parse the table structure from the OCR text
#     headers, rows = parse_table(extracted_text)
    
#     # Organize rows into a DataFrame
#     data = {headers[0]: [], headers[1]: []}
#     for row in rows:
#         if len(row) >= 2:
#             data[headers[0]].append(row[0])
#             data[headers[1]].append(" ".join(row[1:]))
    
#     df = pd.DataFrame(data)
    
#     # Save to Excel
#     latest_excel_file_path = os.path.join(UPLOAD_DIR, "Text_extracted_data.xlsx")
#     df.to_excel(latest_excel_file_path, index=False, engine='openpyxl')
    
#     return JSONResponse(content={
#         "extracted_data": extracted_text,
#         "excel_file": latest_excel_file_path
#     })

# @app.get("/download-excel")
# async def download_excel():
#     # Download the latest Excel file saved
#     if not latest_excel_file_path:
#         return JSONResponse(content={"error": "No file found to download."}, status_code=404)
#     return FileResponse(latest_excel_file_path, filename="Text_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
import cv2
import pytesseract
import os
import numpy as np
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from fuzzywuzzy import process
from textblob import TextBlob

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Waleed Ansari\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

app = FastAPI()

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define an empty string to hold the latest Excel file path
latest_excel_file_path = ""

correct_words = ["Invoice", "Street", "Quantity", "Amount", "Total", "Discount", "Security", "Automation", "Solar", "Panel"]
BRIGHTNESS = 1.0
GAMMA = 2.0
SATURATION = 2.0

def adjust_brightness_contrast(img, brightness=50, contrast=30):
    adjusted = cv2.convertScaleAbs(img, alpha=contrast/127 + 1, beta=brightness - contrast)
    return adjusted

def gamma_correction(img, gamma=1.5):
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, look_up_table)

def adjust_saturation(img, saturation=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation  
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
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
    
    # Ensure the extracted text is a string (in case of list or multiple lines, join them)
    if isinstance(extracted_text, list):
        extracted_text = " ".join(extracted_text)  # Join the list into a single string
    elif not isinstance(extracted_text, str):
        extracted_text = str(extracted_text)  # Ensure it's a string if it's some other type
    
    matched_keywords = fuzzy_match_keywords(extracted_text, correct_words)
    return extracted_text, matched_keywords

def parse_table(extracted_text):
    # Ensure extracted_text is a string and split it into lines
    lines = extracted_text.split('\n')
    
    # Possible headers for the table
    possible_headers = ["Description", "Quantity", "Price", "Amount"]
    headers = []
    
    # Try to dynamically detect the headers
    for line in lines:
        line = line.strip()
        if any(header in line for header in possible_headers):
            headers = [word for word in line.split() if word in possible_headers]
            if len(headers) == 4:
                break
    
    if not headers:
        headers = ["Description", "Quantity", "Price", "Amount"]  # Default headers if no match is found
    
    rows = []
    for line in lines:
        if line.strip():  # Avoid empty lines
            parts = line.split()
            if len(parts) >= len(headers):  # Ensure there are at least as many parts as headers
                row = {headers[i]: parts[i] for i in range(len(headers))}
                rows.append(row)

    return headers, rows

@app.post("/extract-text/")
async def extract_text_from_image(file: UploadFile = File(...)):
    global latest_excel_file_path  # Use global to track the latest file path
    image_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    extracted_text, matched_keywords = extract_text(image_path)
    
    # Parse the table structure from the OCR text
    headers, rows = parse_table(extracted_text)
    
    # Organize rows into a DataFrame
    data = {header: [] for header in headers}
    for row in rows:
        for header in headers:
            data[header].append(row.get(header, ""))
    
    df = pd.DataFrame(data)
    
    # Save to Excel
    latest_excel_file_path = os.path.join(UPLOAD_DIR, "Text_extracted_data.xlsx")
    df.to_excel(latest_excel_file_path, index=False, engine='openpyxl')
    
    return JSONResponse(content={
        "extracted_data": extracted_text,
        "excel_file": latest_excel_file_path
    })

@app.get("/download-excel")
async def download_excel():
    # Download the latest Excel file saved
    if not latest_excel_file_path:
        return JSONResponse(content={"error": "No file found to download."}, status_code=404)
    return FileResponse(latest_excel_file_path, filename="Text_extracted_data.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
