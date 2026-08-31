import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Download model from Hugging Face
MODEL_PATH = hf_hub_download(
    repo_id="SambitPrusty04/rice-cnn-model",
    filename="rice_cnn_model.h5"   # adjust if filename differs
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Class names
classes = ["Basmati", "Ipsala", "Jasmine"]

# Page config with rice icon
st.set_page_config(
    page_title="🌾 Rice Classifier",
    page_icon="🍚",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom background color using CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fdf6e3; /* soft rice-like beige */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with emoji
st.title("🍚 Rice Classification using CNN 🌾")

uploaded_file = st.file_uploader(
    "Upload a Rice Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img = np.array(img)

    if len(img.shape) == 2:  # grayscale
        img = np.stack((img,) * 3, axis=-1)
    if img.shape[-1] == 4:  # RGBA
        img = img[..., :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("🔍 Classifying..."):
        prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🌾 Predicted Rice Type: **{predicted_class}**")
    st.write(f"📊 Confidence: {confidence:.2f}%")

    # Show probability chart
    import pandas as pd
    probs = prediction[0]
    df = pd.DataFrame({"Class": classes, "Probability": probs})
    st.bar_chart(df.set_index("Class"))
