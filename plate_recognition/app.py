from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pytesseract
import easyocr
from werkzeug.utils import secure_filename
from yolov8_utils import YOLOv8Detector
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

detector = YOLOv8Detector(model_path='yolov8s.pt', device='cpu')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_plate_region_yolo(img):
    boxes = detector.detect(img)
    if not boxes:
        return None
    # Assuming the first detected box is the plate
    box = boxes[0]['box']
    x1, y1, x2, y2 = box
    plate_img = img[y1:y2, x1:x2]
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
    plate_img = detect_plate_region_yolo(img)
    if plate_img is not None:
        processed_img = enhance_image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY))
    else:
        processed_img = enhance_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Use EasyOCR for better OCR results with bounding boxes
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(processed_img)

    # Create a color image to draw bounding boxes
    color_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes around each detected character/block
    for (bbox, text, prob) in result:
        # bbox is a list of 4 points
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(color_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Save the image with bounding boxes
    boxed_filename = "boxed_" + os.path.basename(image_path)
    boxed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], boxed_filename)
    cv2.imwrite(boxed_filepath, color_img)

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

    return combined_text, boxed_filename

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
            plate_text, boxed_filename = process_image(filepath)
            return render_template('result.html', filename=boxed_filename, plate_text=plate_text)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
    