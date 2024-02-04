# import streamlit as st
import os
import time
import pandas  as pd
import pickle
import nltk
import spacy
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
# from streamlit_option_menu import option_menu
# from streamlit_extras.switch_page_button import switch_page
# from streamlit_card import card
from PIL import Image
# import tensorflow as tf
import keras
import cv2
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import joblib
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras import Sequential
from keras.layers import Dense, Flatten
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image

from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import gc


# import streamlit as st
import os
import time
import pandas  as pd
import pickle
import nltk
import spacy
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
# from streamlit_option_menu import option_menu
# from streamlit_extras.switch_page_button import switch_page
# from streamlit_card import card
from numpy.linalg import norm
from PIL import Image
import tensorflow as tf
import keras
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras import Sequential
from keras.layers import Dense, Flatten
from numpy.linalg import norm



from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import gc

from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras import Sequential

from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, Flatten
import cv2
import os
import random
from tqdm import tqdm
import numpy as np
from keras.preprocessing import image
import os
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import streamlit as st
import shutil


from flask import Flask, jsonify, request, render_template
import requests, pickle

from recommend_from_images import recommend_images_single

from recommend_from_images_multi import recommend_images_multi

from recommend_from_drawings import recommend_images_drawings

from recommend_from_sketches import recommend_images_sketches

from wardrobe_ai import recommend_wardrobe

from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/')
def home():
    return 'Home'



@app.route('/imageSingle', methods=['GET', 'POST'])
def recommend_from_image_single():
    
    full_path = request.full_path
    print(full_path)
    query_parameter = full_path.split('query=')[1]
    print(query_parameter)
    
    json_object = recommend_images_single(query_parameter)
    print(json_object)
    return json_object
    
@app.route('/imageMulti', methods=['GET', 'POST'])
def recommend_from_image_multi():
    
    full_path = request.full_path
    print(full_path)
    query_parameter = full_path.split('query=')[1]
    print(query_parameter)
    
    json_object = recommend_images_multi(query_parameter)
    
    return json_object
    

@app.route('/sketches', methods=['GET', 'POST'])
def recommend_from_sketches():
    
    full_path = request.full_path
    print(full_path)
    query_parameter = full_path.split('query=')[1]
    print(query_parameter)
    
    json_object = recommend_images_sketches(query_parameter)
    
    return json_object 


@app.route('/drawings', methods=['GET', 'POST'])
def recommend_from_drawings():
    
    full_path = request.full_path
    print(full_path)
    query_parameter = full_path.split('query=')[1]
    print(query_parameter)
    
    json_object = recommend_images_drawings(query_parameter)
    
    return json_object 



@app.route('/ai', methods=['GET', 'POST'])
def recommend_from_wardrobe_ai():
    
    full_path = request.full_path
    print(full_path)
    query_parameter = full_path.split('query=')[1]
    print(query_parameter)
    
    json_object = recommend_wardrobe(query_parameter)
    
    return json_object 


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)