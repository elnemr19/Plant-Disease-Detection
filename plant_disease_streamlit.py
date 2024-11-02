import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st


import tensorflow as tf
from PIL import Image
import json





with open("labels.json", "r") as f:
    labels = json.load(f)


st.title('Plant Disease Prediction')



cnn_model = tf.keras.models.load_model("CNN_model.h5")
efficientnet_model = tf.keras.models.load_model("EfficientNet_model.h5")
vgg19_model = tf.keras.models.load_model("VGG19_model.h5")
resnet_model = tf.keras.models.load_model("ResNet.h5")



# Function to preprocess image
def preprocess_image(image, target_size=(100, 100)):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Convert to RGB to ensure 3 channels
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Prediction function
def predict_image(image_array, model):
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # For multi-class
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Streamlit UI
#st.title("Image Classification App")
st.write("Upload an image to classify.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Model selection
model_choice = st.selectbox("Choose a model", ["CNN", "EfficientNet", "VGG19", "ResNet"])

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    image_array = preprocess_image(image)
    
    # Choose model
    if model_choice == "CNN":
        model = cnn_model
    elif model_choice == "EfficientNet":
        model = efficientnet_model
    elif model_choice == "VGG19":
        model = vgg19_model
    elif model_choice == "ResNet":
        model = resnet_model
    
    # Predict
    predicted_class, confidence = predict_image(image_array, model)
    st.write(f"Prediction: Class {labels[predicted_class]},                       Confidence: {confidence:.2f}%")



