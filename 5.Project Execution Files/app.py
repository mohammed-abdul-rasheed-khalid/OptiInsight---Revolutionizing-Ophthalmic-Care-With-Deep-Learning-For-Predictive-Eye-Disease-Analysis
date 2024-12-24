import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from PIL import Image

# Load the model
try:
    model = load_model("evgg.keras")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle gracefully if model loading fails

# Initialize Flask app
app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the predict page
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            if not model:
                return "Model is not loaded. Please check the server logs for details."

            # Retrieve the uploaded file
            f = request.files.get('image', None)
            if not f:
                return "No file uploaded. Please upload an image file."
            
            # Validate file type
            if not f.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return "Invalid file format. Please upload a .png, .jpg, or .jpeg file."
            
            # Save the uploaded file
            filepath = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(filepath)
            
            # Preprocess the image
            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            img_data = preprocess_input(x)
            
            # Make the prediction
            prediction = np.argmax(model.predict(img_data), axis=1)
            index = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
            result = str(index[prediction[0]])
            
            # Render the output page
            return render_template('output.html', prediction=result)
        except Exception as e:
            print(f"Error: {e}")
            return f"An error occurred: {e}"
    
    # Render the upload form when the request method is GET
    return render_template('predict.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
