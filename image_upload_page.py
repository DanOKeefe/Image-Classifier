import streamlit as st
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.mobilenet import preprocess_input

@st.cache()
def load_model():
    model =  tf.keras.models.load_model('mobilenet')
    model.compile()
    return model

def process_image(img):
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return preprocess_input(img_arr)

def app():
    model = load_model()
    st.markdown(
        f"<h1 style='text-align: center;'>Image Classifier</h1>", 
        unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Choose a file')
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        img_arr = process_image(img)
        y_pred = model.predict(img_arr)
        results = imagenet_utils.decode_predictions(y_pred)
        image_caption = f'Prediction: {results[0][0][1]}'
        st.image(img, caption=image_caption)
    
    
