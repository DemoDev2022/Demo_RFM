import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
model = keras.models.load_model('E:/KMUTT/SED 630 Introduction to Neural Network and Deep Learning/Deploy/model')
# model = keras.models.load_model('model')

st.title( 'Arabic number recognizer')
st.write( 'Web app for handwritten digit recognition')

SIZE = 192
canvas_result = st_canvas(
    fill_color= '#000000',
    stroke_width= 20,
    stroke_color='#FFFFFF',
    background_color= '#000000',
    width=SIZE, 
    height=SIZE,drawing_mode="freedraw",
    key = 'canvas'
)

if st.button('Predict'):
     img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28) )
     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     img = img.astype('float32')
     img = img.reshape(1, 28, 28, 1)
     img /= 255
     digit = model.predict(img)
     classes=np.argmax(digit(0))
     st.write(f'Predicted Result:{classes}')
