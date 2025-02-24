'''import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource  # Cache model to avoid reloading on every interaction
def load_model():

    model = tf.keras.models.load_model("skin_lesion_model (1).keras")
    return model

model = load_model()

# Class labels (same as used during training)
class_labels = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7']  # Replace with actual class names

# Image preprocessing function
# Image preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")  # Convert RGBA/Grayscale to RGB
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array.astype(np.float32)  # Ensure correct data type


# Streamlit UI
st.title("Skin Lesion Classification")
st.write("Upload an image to classify the skin lesion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]  # Get class with highest probability

    # Display results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {np.max(prediction) * 100:.2f}%")'''

from huggingface_hub import hf_hub_download
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model from Hugging Face
@st.cache_resource  # Cache model to avoid reloading
def load_model():
    model_path = hf_hub_download(repo_id="puji-poojitha/skin-pred-101", filename="skin_lesion_model (1).keras")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Class labels (Replace with actual disease names)
class_labels = ["Melanoma", "Nevus", "Basal Cell Carcinoma", "Actinic Keratosis", "Dermatofibroma", "Vascular Lesion", "Benign Keratosis"]

# Image preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")  # Convert RGBA/Grayscale to RGB
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array.astype(np.float32)  # Ensure correct data type

# Streamlit UI
st.title("Skin Lesion Classification")
st.write("Upload an image to classify the skin lesion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]  # Get class with highest probability

    # Display results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {np.max(prediction) * 100:.2f}%")
