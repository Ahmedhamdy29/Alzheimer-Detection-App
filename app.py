import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# تحميل الموديل
# -----------------------------
model = tf.keras.models.load_model("best_model_f.h5")
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Alzheimer's MRI Classification")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    img = image.resize((224,224))
    img_array = np.expand_dims(np.array(img), axis=0)

    # Prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)

    st.write(f"### 🏷 Prediction: {class_names[class_idx]} ({confidence*100:.2f}%)")

    # Show probabilities
    st.write("### Class Probabilities")
    for i, prob in enumerate(preds[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")

