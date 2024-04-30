import sys
import pkg_resources
import subprocess

def install_required_libraries():

    def install(package):
        python = sys.executable
        print(f"Install {package}...")
        subprocess.check_call([python, '-m', 'pip', 'install', package])
        print(f"Install successfully\n----------------------")

    requirements = None
    with open(r"options/requirements.txt") as f:
        requirements = f.read()
     
    requirements = requirements.split("\n")
    installed = [pkg.key for pkg in pkg_resources.working_set]

    for package in requirements:
        if package not in installed:
            install(package)

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
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE)
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

from . import make_celery
from flask import Flask, jsonify
from celery import Celery
from .config import Config
import tensorflow.keras.applications.mobilenet_v2 as mobilenet
import tensorflow as tf
import numpy as np
import io
import base64
import cv2 as cv
import gdown
import os.path
import PIL

IMAGE_SIZE = (128, 128, 3)
map_classes = {
    0: "airplane", 1: "bicycle", 2: "boat", 3: "bus",
    4: "car", 5: "motorcycle", 6: "train", 7: "truck"
}

url = 'https://drive.google.com/uc?id=1sZBWSTma0pD0JkpXT0HExMcYGe7LQFdv'
output = r"weights.hdf5"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
    
conv_NN = get_compiled_model()
conv_NN.load_weights(output)

celery = make_celery()

@celery.task(name="NN.predict")
def predict(data):
    images = data
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
    
    return answers