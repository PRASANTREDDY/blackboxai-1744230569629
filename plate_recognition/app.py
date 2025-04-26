from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pytesseract
import easyocr
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_plate_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter to preserve edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    edged = cv2.Canny(filtered, 30, 200)
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break
    if plate_contour is None:
        return None
    # Create mask and extract plate
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_img = gray[y:y+h, x:x+w]
    return plate_img

def enhance_image(img):
    # Resize image to fixed width while maintaining aspect ratio
    fixed_width = 400
    height = int(img.shape[0] * (fixed_width / img.shape[1]))
    img = cv2.resize(img, (fixed_width, height), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(sharpened)
    # Morphological opening to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_open)
    # Adaptive thresholding for better binarization
    thresh = cv2.adaptiveThreshold(opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def process_image(image_path):
    img = cv2.imread(image_path)
    plate_img = detect_plate_region(img)
    if plate_img is not None:
        processed_img = enhance_image(plate_img)
    else:
        processed_img = enhance_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Use EasyOCR for better OCR results
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(processed_img)

    # Extract text from EasyOCR result
    texts = [res[1] for res in result]
    combined_text = ' '.join(texts).strip()

    # If EasyOCR fails, fallback to pytesseract
    if not combined_text:
        psm_modes = [3, 6, 7, 8]
        ocr_results = []
        for psm in psm_modes:
            custom_config = f'--oem 3 --psm {psm}'
            text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
            if text:
                ocr_results.append(text)
        if ocr_results:
            combined_text = max(ocr_results, key=len)
        else:
            combined_text = ""

    return combined_text

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            plate_text = process_image(filepath)
            return render_template('result.html', filename=filename, plate_text=plate_text)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
    