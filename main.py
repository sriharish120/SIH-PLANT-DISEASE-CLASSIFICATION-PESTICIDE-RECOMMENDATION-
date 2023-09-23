import streamlit as st
import tensorflow as tf2
import numpy as np
from PIL import Image
import tensorflow as tf2
import tensorflow.compat.v2 as tf2
import keras
import pandas as pd
st.markdown('<h1 style="color:white;">Plant disease Classification</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into 38 classes.</h2>', unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)
df = pd.read_csv("C:\\Users\\SRIHARISH\\Downloads\\predictions.csv", encoding='ISO-8859-1')
df1 = df.iloc[:, :-2]
@st.cache(allow_output_mutation=True)
def load_model():
	model = tf2.keras.models.load_model("C:\\Users\\SRIHARISH\\Downloads\\modelVGG19.h5")
	return model


def predict_class(image, model):

	image = tf2.cast(image, tf2.float32)
	image = tf2.image.resize(image, [224, 224])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Classifier')

file = st.file_uploader("Upload an image of the leaves", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
	#formatted = f"{class_names:,d}"

	result = class_names[np.argmax(pred)]
	st.markdown('<p class="big-font">prediction</p>', unsafe_allow_html=True)
	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)
	st.markdown('<p class="big-font">Recommendation:</p>', unsafe_allow_html=True)
	filtered_df = df1[df1['predictions'] == result]
	if not filtered_df.empty:
		solution = filtered_df['solutions'].iloc[0]
		st.write("Solution for", result,":", solution)
	else:
		st.write("prediction not found in the DataFrame.")