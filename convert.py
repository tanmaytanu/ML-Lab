import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import os

# Set the path to the Tesseract executable (if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def image_to_text(image_path):
    """Extract text from an image using OCR."""
    return pytesseract.image_to_string(image_path)

def pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text

def save_text_to_csv(text, csv_path):
    """Save extracted text to a CSV file."""
    # Split text into lines and save as a CSV
    lines = text.split("\n")
    df = pd.DataFrame(lines, columns=["Text"])
    df.to_csv(csv_path, index=False)

def convert_to_csv(input_path, output_path):
    """Convert an image or PDF file to a CSV file."""
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        print("Processing image file...")
        text = image_to_text(input_path)
    elif input_path.lower().endswith('.pdf'):
        print("Processing PDF file...")
        text = pdf_to_text(input_path)
    else:
        raise ValueError("Unsupported file format. Please provide an image or PDF file.")

    save_text_to_csv(text, output_path)
    print(f"Text extracted and saved to {output_path}")

# Example usage
input_file = "/home/tanmay/Documents/csv/notice.jpeg"  # Replace with your file path
output_file = "/home/tanmay/Documents/csv/notice.csv"  # Replace with your desired output CSV file path

convert_to_csv(input_file, output_file)
