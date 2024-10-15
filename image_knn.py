import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['DATASET_FOLDER'] = 'dataset'  
app.config['UPLOAD_FOLDER'] = 'uploads/'  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to extract image features
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  
    features = image.flatten()  
    return features

# Load images and extract features
dataset_path = './dataset'  
images = []
features_list = []

for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(dataset_path, filename)
        features = extract_features(img_path)
        features_list.append(features)
        images.append(filename)

print(f"Number of images loaded: {len(images)}")  # Check number of images

# Normalize features
scaler = StandardScaler()
features_list_normalized = scaler.fit_transform(features_list)

# Create KNN model
n_neighbors = min(2, len(features_list))  # Adjust number of neighbors
knn = NearestNeighbors(n_neighbors=n_neighbors)
knn.fit(features_list_normalized)

# Function to find similar images
def find_similar_images(input_image_path):
    input_features = extract_features(input_image_path)
    input_features_normalized = scaler.transform([input_features])  # Normalize input image
    distances, indices = knn.kneighbors(input_features_normalized)

    similar_images = []
    threshold_distance = 1000  # Adjust this value based on results
    for i, index in enumerate(indices[0]):
        distance = distances[0][i]
        if images[index] != os.path.basename(input_image_path) and distance < threshold_distance:
            similar_images.append(images[index])

    return similar_images

# Route to serve images from the dataset
@app.route('/dataset/<path:filename>')
def dataset_file(filename):
    return send_from_directory(app.config['DATASET_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        similar_images = find_similar_images(file_path)
        
        if not similar_images:
            return render_template('results.html', similar_images=None)  # No similar images
        else:
            return render_template('results.html', similar_images=similar_images)  # Found images

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
