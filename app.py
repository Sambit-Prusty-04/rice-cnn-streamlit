import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("rice_cnn_model.h5")

# Class names
classes = ["Basmati", "Ipsala", "Jasmine"]

# Title
st.title("Rice Classification using CNN")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Rice Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    img = image.resize((128,128))
    img = np.array(img)

    # If image is grayscale convert to RGB
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]

    st.success(
        f"Predicted Rice Type: {predicted_class}"
    )