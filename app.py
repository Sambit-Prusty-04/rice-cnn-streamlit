import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Download model from Hugging Face
MODEL_PATH = hf_hub_download(
    repo_id="SambitPrusty04/rice-cnn-model",
    filename="rice_cnn_model.h5"   # change if your filename is different
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
classes = ["Basmati", "Ipsala", "Jasmine"]

# Title
st.title("Rice Classification using CNN")

uploaded_file = st.file_uploader(
    "Upload Rice Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    img = image.resize((128, 128))
    img = np.array(img)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(
        f"Predicted Rice Type: {predicted_class}"
    )

    st.write(f"Confidence: {confidence:.2f}%")