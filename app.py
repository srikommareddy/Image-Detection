#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cifar10_cnn_model.h5')

# Define the class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image):
    """
    Preprocess the uploaded image to the required format for the model.
    """
    image = ImageOps.fit(image, (32, 32), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_image(image, model):
    """
    Classify the preprocessed image using the loaded model.
    """
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[predicted_class], confidence

# Streamlit app layout
st.title("CIFAR-10 Image Classification")
st.write("Upload an image to classify it into one of the CIFAR-10 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and classify the image
    preprocessed_image = preprocess_image(image)
    predicted_class, confidence = classify_image(preprocessed_image, model)

    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
