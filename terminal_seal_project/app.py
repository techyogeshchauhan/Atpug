import os
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, request, send_file, jsonify, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil
import zipfile
from datetime import datetime
import random
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DETECTED_FOLDER'] = 'static/detected'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize YOLO model
MODEL_PATH = 'D:/Atpug/terminal_seal_project/runs/train/exp/best.pt'
model = YOLO(MODEL_PATH)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

def generate_meter_id():
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    rand = random.randint(1000, 9999)
    return f"MTR{now}{rand}"

def detect_seals(img_path, conf=0.1):
    """Detect terminal seals in an image and save result. Returns (output_filename, detection_count)"""
    results = model.predict(img_path, conf=conf, save=False)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detection_count = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = box.conf[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Seal: {conf_score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detection_count += 1
    output_filename = f"detected_{Path(img_path).name}"
    output_path = os.path.join(app.config['DETECTED_FOLDER'], output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return output_filename, detection_count

def save_to_excel(entries, excel_path):
    # entries: list of dicts with keys: meter_id, filename, seal_available
    df_new = pd.DataFrame(entries)
    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_excel(excel_path, index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    detection_table = []
    excel_path = os.path.join(app.config['DETECTED_FOLDER'], 'detection_summary.xlsx')
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
            detection_table = df.to_dict(orient='records')
        except Exception as e:
            detection_table = []
    if request.method == 'POST':
        conf = float(request.form.get('conf', 0.1))
        # excel_path already defined above
        # Handle single image upload
        if 'image' in request.files and request.files['image'].filename:
            file = request.files['image']
            if file.filename.endswith(('.jpg', '.png')):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)
                # Detect seals
                detected_filename, detection_count = detect_seals(upload_path, conf)
                meter_id = generate_meter_id()
                seal_available = detection_count > 0
                # Save to Excel
                entry = [{
                    'meter_id': meter_id,
                    'filename': detected_filename,
                    'seal_available': 'Available' if seal_available else 'Not Available'
                }]
                save_to_excel(entry, excel_path)
                # Update detection_table after saving
                try:
                    df = pd.read_excel(excel_path)
                    detection_table = df.to_dict(orient='records')
                except Exception as e:
                    detection_table = []
                return render_template('index.html', detected_image=detected_filename, conf=conf, meter_id=meter_id, seal_available=seal_available, detection_table=detection_table)
        # Handle folder upload (zip file)
        if 'folder' in request.files and request.files['folder'].filename:
            file = request.files['folder']
            if file.filename.endswith('.zip'):
                zip_filename = secure_filename(file.filename)
                zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
                file.save(zip_path)
                # Extract zip
                extract_path = os.path.join(app.config['UPLOAD_FOLDER'], Path(zip_filename).stem)
                os.makedirs(extract_path, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                except zipfile.BadZipFile:
                    shutil.rmtree(extract_path, ignore_errors=True)
                    os.remove(zip_path)
                    return render_template('index.html', error="Uploaded zip file is corrupt.", conf=conf, detection_table=detection_table)
                # Process images
                detected_images = []
                detected_info = []
                excel_entries = []
                image_found = False
                for img_path in Path(extract_path).rglob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        image_found = True
                        detected_filename, detection_count = detect_seals(str(img_path), conf)
                        meter_id = generate_meter_id()
                        seal_available = detection_count > 0
                        detected_images.append(detected_filename)
                        detected_info.append({
                            'filename': detected_filename,
                            'meter_id': meter_id,
                            'seal_available': seal_available
                        })
                        excel_entries.append({
                            'meter_id': meter_id,
                            'filename': detected_filename,
                            'seal_available': 'Available' if seal_available else 'Not Available'
                        })
                if not image_found:
                    shutil.rmtree(extract_path, ignore_errors=True)
                    os.remove(zip_path)
                    return render_template('index.html', error="No images (.jpg/.jpeg/.png) found in the uploaded zip.", conf=conf, detection_table=detection_table)
                # Save all to Excel
                save_to_excel(excel_entries, excel_path)
                # Update detection_table after saving
                try:
                    df = pd.read_excel(excel_path)
                    detection_table = df.to_dict(orient='records')
                except Exception as e:
                    detection_table = []
                # Create zip of detected images
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                zip_output = os.path.join(app.config['DETECTED_FOLDER'], f'detected_{timestamp}.zip')
                with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for img in detected_images:
                        detected_img_path = os.path.join(app.config['DETECTED_FOLDER'], img)
                        if os.path.exists(detected_img_path):
                            zipf.write(detected_img_path, img)
                # Clean up extracted folder and uploaded zip
                shutil.rmtree(extract_path, ignore_errors=True)
                os.remove(zip_path)
                return render_template('index.html', detected_images=detected_images, zip_file=Path(zip_output).name, conf=conf, detected_info=detected_info, detection_table=detection_table)
        return render_template('index.html', error="Invalid file format. Use .jpg/.png for images or .zip for folders.", conf=conf, detection_table=detection_table)
    # On GET (refresh), do not show detected image/info
    return render_template('index.html', conf=0.1, detection_table=detection_table)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['DETECTED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found!")
        exit(1)
    app.run(debug=True, host='0.0.0.0', port=5000)