from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
from detect import annotate_image
import cv2
import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['ANNOTATED_IMAGES_FOLDER'] = 'annotated_images'
annotations = []
# Create the annotated images folder if it doesn't exist
annotated_images_folder = os.path.join(app.root_path, app.config['ANNOTATED_IMAGES_FOLDER'])
os.makedirs(annotated_images_folder, exist_ok=True)

def load_annotations():
    annotations_file = os.path.join(app.root_path, 'annotations.json')
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            return json.load(f)
    else:
        return []
    

def save_annotations(annotations):
    annotations_file = os.path.join(app.root_path, 'annotations.json')
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle image upload and perform license plate recognition
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Perform license plate recognition on the uploaded image
    annotated_image, label = annotate_image(file_path)
    
    # Save the annotated image with a unique filename
    annotated_image_filename = f"{os.path.splitext(file.filename)[0]}_annotated.jpg"
    annotated_image_path = os.path.join(app.config['ANNOTATED_IMAGES_FOLDER'], annotated_image_filename)
    cv2.imwrite(annotated_image_path, annotated_image)
    # Store the annotation information
    annotation = {'image_filename': annotated_image_filename, 'label': label}
    annotations = load_annotations()
    annotations.append(annotation)
    save_annotations(annotations)
    annotations = load_annotations()
    # Pass the filename of the annotated image and label to the template
    return render_template('results.html', image_filename=annotated_image_filename, label=label, annotations=annotations)


@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(app.config['STATIC_FOLDER'], path)


@app.route('/annotated_images/<path:path>')
def annotated_images(path):
    return send_from_directory(app.config['ANNOTATED_IMAGES_FOLDER'], path)


if __name__ == '__main__':
    app.run(debug=True)
