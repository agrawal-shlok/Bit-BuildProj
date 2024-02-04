import streamlit as st
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
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
from PIL import Image
import tensorflow as tf
import keras
import cv2
from numpy.linalg import norm
from keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import joblib
from keras.layers import GlobalMaxPooling2D
from keras import Sequential
from keras.layers import Dense, Flatten
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from keras.preprocessing import image

from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import gc


import firebase_admin
from firebase_admin import credentials, storage


import streamlit as st
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
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
from numpy.linalg import norm
from PIL import Image
import tensorflow as tf
import keras
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import preprocess_input
from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from keras.layers import GlobalMaxPooling2D
from keras import Sequential
from keras.layers import Dense, Flatten
from numpy.linalg import norm

import uuid

from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import gc
import uuid

from keras.layers import Input, Dense, Conv2D
from keras import Sequential

from keras.models import Model
from keras.layers import UpSampling2D, MaxPooling2D, Flatten
import cv2
import os
import random
from tqdm import tqdm
import numpy as np
from keras.preprocessing import image
import os
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st
import shutil


from flask import Flask, jsonify, request, render_template
import requests, pickle


from firebase_admin import credentials, storage, initialize_app


service_account_key_path = 'serviceAccountKey.json'


cred = credentials.Certificate(service_account_key_path)
firebase_app = initialize_app(cred, {'storageBucket': 'fashionx-ebe6c.appspot.com'}, name="drawings")



firebase_admin.initialize_app(cred, {'storageBucket': 'fashionx-ebe6c.appspot.com'})

def recommend_images_drawings(filename):
    
    def vgg():
        vgg16 = VGG16(weights='imagenet', input_shape = (224,224,3), include_top=False)
        for layers in vgg16.layers:
            layers.trainable=False
        
        return vgg16
    def features_image_drawings():
        features_images = pickle.load(open('embeddings_images_20000_recommend_drawings_vgg16.pkl', 'rb'))
        return features_images      
        
    filenames_drawings = pickle.load(open('images_recommend_20000_filenames_drawings.pkl', 'rb'))
    
  
    
    
    def extract(img_path, vgg16):
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        expanded_img = np.expand_dims(img_array,axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        
        result = vgg16.predict(preprocessed_img, verbose=0).flatten()
        return result
    
    def get_download_url(file_path):
   
            storage_ref = storage.bucket()
            print(file_path.split('/')[1])
            # Specify the file path within the storage b    ucket
            blob = storage_ref.blob(file_path.split('/')[1])
            print(blob)
            # Get the downloadable URL for the file
            download_url = blob.generate_signed_url(expiration=36000000000) # Set expiration time (in seconds)

            return download_url


    ls = []

    index_pos_shirts = []
    index_pos_pants = []
    index_pos_shoes = []
    index_pos_shorts = []
    # filenames_drawings = []
    index_pos = []

    # if st.checkbox('Upload a drawing/outline'):
    # if os.path.exists('uploads2/'):
    #     shutil.rmtree('uploads2/')
    #     os.mkdir('uploads2/')
    # # else:
    # else:
    os.makedirs('uploads2/', exist_ok=True)

    
    uid = uuid.uuid1()
    with open('uploads2/{}.png'.format(uid), 'wb') as f:
        url = requests.get(filename)
        
        # body = url.body
        f.write(url.content)
        
    # uploaded_image = st.file_uploader('Upload an outline/drawing')

    # if uploaded_image is not None:

    #     if save_uploaded_image_outline(uploaded_image):
        # if save_uploaded_image(uploaded_image):
        #     if os.path.exists('results_sketches'):
        #         shutil.rmtree('results_sketches')
            # else:
            #     os.mkdir('results_sketches')

            # model.predict(os.path.join('uploads/', uploaded_image.name), save=True, save_txt=True, save_crop=True, project='results_sketches')


    #         for file in os.listdir('results_hed/predict/crops/'):

    #             if file in ['shirt', 'jacket', 'dress']:

    #                 file2 = 'shirt'

    #                 os.rename('results_hed/predict/crops/{}/'.format(file), 'results_hed/predict/crops/{}/'.format(file2))

    #         for file in os.listdir('uploads2/'):

    #            # print(file)
        similarity=[]
        features_images_drawings = features_image_drawings()
        # if st.checkbox('Recommend'):
        #     for file in os.listdir('results/predict/crops/'):
        similarity = []
        # if st.checkbox('Recommend'):
    for img in os.listdir('uploads2/'):

        vgg16 = vgg()
        # gc.collect()
        features = extract('uploads2/' + img, vgg16)

        for i in range(len(features_images_drawings)):

            similarity.append((cosine_similarity(features.reshape(1,-1), features_images_drawings[i].reshape(1, -1))))

        similarity = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
        # print(similarity)

        # if st.checkbox('Show'):
        # for file in os.listdir('uploads2/'):
#                         if file == 'short':
                # with st.expander('Top 10 recommndations'):
#                                 if len(index_pos_shorts) != 0:

                    # columns = st.columns(10)
        # print(similarity)
        index = []
        for i in range(10):
                # print(similarity[i][0])
                
            index.append(similarity[i][0])
        filenames = []
        # print(filenames_drawings)
        for i in range(10):
            # with columns[i]:
                # print(filenames_drawings[index[i]])
                # temp = ' '.join(filenames_drawings[index[i]].split('/')[-1:])
                # temp = temp.split('.')[-2]
                temp = filenames[index_pos_pants[i]].split('/')[-1]
                # print(temp)
                # print(filenames[similarity[0][0]].split('/')[-1:])
                # path='/'.join(filenames_drawings[similarity[i][0]].split('/')[-1:])
                # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                print('images/' + temp + '.jpg')
                # image = cv2.imread(('images/' + temp + '.jpg'))
                # image = cv2.resize(image, (224, 224))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # st.write(final_df.iloc[index_pos_shirts[i],4])

                # st.image(image)
                filenames.append('images/' + temp)

            # else:
        #         st.write('None')
        
        urls = []
    
        
        for i in filenames:
            url = get_download_url(i)
            urls.append(url)
        
    return jsonify({
        "status": "success",
        "prediction": urls
        # "confidence": str(classes[0][0][2]),
        # "upload_time": datetime.now()
    })        #     st.write('None')
