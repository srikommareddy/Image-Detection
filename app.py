#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('cifar10_model.h5')  # Update with the path to your saved model

# Define the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classification")
st.write("Upload an image to classify it into one of the 10 CIFAR-10 classes.")

# File uploader allows user to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file)
    
    # Display the image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image
    img = img.resize((32, 32))  # CIFAR-10 images are 32x32 pixels
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class of the image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    st.write(f"Predicted class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")

