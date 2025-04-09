# Vehicle Number Plate Recognition System

This application recognizes characters from vehicle number plates using OpenCV and Tesseract OCR.

## Prerequisites
- Python 3.7+
- Tesseract OCR installed on your system (https://github.com/tesseract-ocr/tesseract)

## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set Tesseract path in your environment (if not in system PATH):
```bash
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

## Running the Application
```bash
python app.py
```

The application will start on http://localhost:5000

## Usage
1. Upload a clear image of a vehicle number plate
2. The system will process the image and display:
   - The original image
   - Recognized text from the number plate

## Notes
- For best results, use clear images of number plates
- The system performs grayscale conversion and thresholding to improve recognition
- You may need to adjust the Tesseract configuration for different number plate formats

## Troubleshooting
If you get errors about Tesseract:
- Ensure Tesseract is installed (`tesseract --version`)
- Set the correct path to Tesseract executable in app.py if needed:
```python
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
