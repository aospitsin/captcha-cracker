### Installing the necessary libraries
import sys
import pkg_resources
import subprocess

def install_required_libraries():
    requirements = None
    with open(r"requirements.txt") as f:
        requirements = f.read()
     
    requirements = requirements.split("\n")
    installed = [pkg.key for pkg in pkg_resources.working_set]

    def install(package):
        python = sys.executable
        print(f"Install {package}...")
        subprocess.check_call([python, '-m', 'pip', 'install', package])
        print(f"Install successfully\n----------------------")

    for r in requirements:
        if not r in installed:
            install(r)

    print("All necessary libraries are installed")

def preprocess_input_model(_image): 
    _image = _image.replace("data:image/jpeg;base64,", "")
    _image = PIL.Image.open(io.BytesIO(base64.b64decode(_image)))
    
    crop_image = _image.crop((20, 30, 110, 90))
    crop_image = crop_image.resize((128, 128), PIL.Image.ANTIALIAS)
    
    crop_image = cv.cvtColor(np.array(crop_image), cv.COLOR_BGRA2BGR)
    
    return mobilenet.preprocess_input(crop_image)[None, ...]

def get_compiled_model():  
    base_model = mobilenet.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(8, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    return model

install_required_libraries()
###

import tensorflow.keras.applications.mobilenet_v2 as mobilenet
import tensorflow as tf
import numpy as np
import io
import base64
import cv2 as cv
import gdown
import os.path
import PIL

from flask import Flask, jsonify, request

IMAGE_SIZE = (128, 128, 3)
map_classes = {
    0: "airplane", 1: "bicycle", 2: "boat", 3: "bus",
    4: "car", 5: "motorcycle", 6: "train", 7: "truck"
}

url = 'https://drive.google.com/uc?id=1sZBWSTma0pD0JkpXT0HExMcYGe7LQFdv'
output = r"weights.hdf5"

path = "tune_model/best_model.hdf5"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

conv_NN = get_compiled_model()
conv_NN.load_weights(output)
# conv_NN.load_weights(path)

app = Flask(__name__)

data = [
    {
        "image": None,
        "answer": None
    }
]

@app.route("/predict", methods=["GET"])
def get_list():
    return jsonify(data)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.data
    images = data
    images = data.decode("utf-8")
    images = images.split("\n")

    answers = ""
    
    for i, base64_image in enumerate(images):
        img = preprocess_input_model(base64_image)
        pred = np.argmax(conv_NN.predict(img), axis=-1)
        
        # Returning the response in the format of a regular list with predictions
        answers += f"{i}~" + map_classes[pred[0]] + "\n"
        
        # Returning a response in JSON format with an image code
        # answer = {
        #     "image": base64_image,
        #     "answer": map_classes[pred[0]]
        # }
        # answers.append(answer)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(threaded=True)