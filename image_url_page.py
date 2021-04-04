import requests
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.mobilenet import preprocess_input

def get_image(img_url):
    return Image.open(requests.get(img_url, stream=True).raw)

def process_image(img):
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return preprocess_input(img_arr)

@st.cache()
def load_model():
    model =  tf.keras.models.load_model('mobilenet')
    model.compile()
    return model

def app():
    model = load_model()
    st.markdown(
        f"<h1 style='text-align: center;'>Image Classifier</h1>", 
        unsafe_allow_html=True)
    img_url = st.text_input(
        label='Enter Image URL',
        value='https://upload.wikimedia.org/wikipedia/commons/e/eb/JenB_Marking_Territory.JPG')
    img = get_image(img_url)
    img_arr = process_image(img)
    y_pred = model.predict(img_arr)
    results = imagenet_utils.decode_predictions(y_pred)
    image_caption = f'Prediction: {results[0][0][1]}'
    st.image(img, caption=image_caption)