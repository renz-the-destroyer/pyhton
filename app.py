from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
 
# flask for web app of python
app = Flask(__name__)
 
# load the trained model
model = tf.keras.models.load_model("model.h5")
 
 
# home page
@app.route("/")
def home():
    return render_template("index.html")
 
 
# prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    img = img.resize((180, 180))
 
    # numpy pixel array, shape (1, 180, 180, 3)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
 
    # sigmoid output: close to 1 = dog, close to 0 = cat
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        result = "DOG"
    else:
        result = "CAT"
 
    # return to HTML page
    return render_template("index.html", prediction=result)
 
 
if __name__ == "__main__":
    app.run(debug=True)