from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(img):
    img = img.resize((28, 28))
    img = img.convert("L")
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img):
    # Load your machine learning model
    model = load_model('models/your_model.h5')
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    class_name = class_names[predicted_class]
    return class_name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'Aucune image téléchargée'
    
    file = request.files['file']
    if file.filename == '':
        return 'Aucun fichier sélectionné'

    if file and allowed_file(file.filename):
        img = Image.open(file)
        img = preprocess_image(img)
        prediction = predict_image(img)
        return 'Le vêtement prédit est : ' + prediction
    else:
        return 'Format de fichier non pris en charge'

if __name__ == '__main__':
    app.run(debug=True, port=8080)
