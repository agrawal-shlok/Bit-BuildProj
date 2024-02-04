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

import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("serviceAccountKeyRealtimeDatabase.json")

try:
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://fashionx-ebe6c-default-rtdb.firebaseio.com/'}, name="ai")
except ValueError as e:
    print(f"Error initializing Firebase app: {e}")
    exit()


from flask import Flask, jsonify, request, render_template
import requests, pickle


def recommend_wardrobe(uid):

    model=tf.keras.saving.load_model('clothes (1).keras')
    
    # if os.path.exists('uploads_shoes/'):
    #     shutil.rmtree('uploads_shoes/')
    #     os.mkdir('uploads_shoes/')

    # else:
    os.makedirs('uploads_shoes/', exist_ok=True)

        
    # if os.path.exists('uploads_pants/'):
    #     shutil.rmtree('uploads_pants/')
    #     os.make('uploads_pants/')

    # else:
    os.makedirs('uploads_pants/', exist_ok=True)
        
        
        
    # if os.path.exists('uploads_outwear/'):
    #     shutil.rmtree('uploads_outwear/')
    #     os.mkdir('uploads_outwear/')

    # else:
    os.makedirs('uploads_outwear/', exist_ok=True)
  
    # Construct URLs
    # print('https://fashionx-ebe6c-default-rtdb.firebaseio.com/' + f'{uid}/WardrobeAI/Outwear')
    # url = 'https://fashionx-ebe6c-default-rtdb.firebaseio.com/' + f'{uid}/WardrobeAI/Outwear.json'
    # response = requests.get(url)
    # print(response.json())
    url_outwear = 'https://fashionx-ebe6c-default-rtdb.firebaseio.com/' + f'{uid}/WardrobeAI/Outwear.json'
    url_legwear = 'https://fashionx-ebe6c-default-rtdb.firebaseio.com/' + f'{uid}/WardrobeAI/Legwear.json'
    url_shoe = 'https://fashionx-ebe6c-default-rtdb.firebaseio.com/' + f'{uid}/WardrobeAI/Shoes.json'

    # Create database references
    # ref_outwear = db.reference(url_outwear)
    # ref_legwear = db.reference(url_legwear)
    # ref_shoe = db.reference(url_shoe)
    
    data_outwear = requests.get(url_outwear).json()
    data_legwear = requests.get(url_legwear).json()
    data_shoe = requests.get(url_shoe).json()
    
    # print(data_legwear)
    # print(data_outwear)
    # print(data_outwear)

    # Now you can use ref_outwear, ref_legwear, and ref_shoe for further operations
    outwear_url = []
    legwear_url = []
    shoes_url = []
    
    for data in data_outwear.values():
        outwear_url.append(data['image_url'])
        
    for data in data_legwear.values():
        legwear_url.append(data['image_url'])
        
    for data in data_shoe.values():
        shoes_url.append(data['image_url'])
    
    print(outwear_url)
    maping = {}
    count = 0
    for i in range(len(outwear_url)):
        with open("uploads_outwear/{}.png".format(count), "wb") as f:
            # print("jkbkjbkjbkjbjk")
            data = requests.get(outwear_url[count])
            # print(data)
            f.write(data.content)
            # print(data.content)
            maping["uploads_outwear/{}.png".format(count)] = outwear_url[count]
            count+=1
    
    count = 0
    for i in range(len(legwear_url)):
        with open("uploads_pants/{}.png".format(count), "wb") as f:
            data = requests.get(legwear_url[count])
            f.write(data.content)
            maping["uploads_pants/{}.png".format(count)] = legwear_url[count]
            count+=1
        
    count = 0
    for i in range(len(shoes_url)):
        with open("uploads_shoes/{}.png".format(count), "wb") as f:
            data = requests.get(shoes_url[count])
            f.write(data.content)
            maping["uploads_shoes/{}.png".format(count)] = shoes_url[count]
            count+=1
        
            
        
    
    

    def get_data_single(folder_path):
    #     temp = []
        temp = []
        
        features = []
        X = []
    
    #         for img in tqdm(os.listdir(folder_path)):
        img = image.load_img(folder_path, target_size=(128, 128, 3))
        img_array = image.img_to_array(img)
        img_array /= 255.
    #             resize_array = cv2.resize(img_array, (128,128))
        temp.append(img_array)

    #     for i in temp:
    #         features.append(i)
    #     X = np.array(temp).reshape(-1, 128, 128, 3)
    #     X = X.astype('float32')
    #     X /= 255
        return np.array(temp).reshape(-1, 128, 128, 3)


    

   
    # def save_uploaded_image_outwear(uploaded_image):
    #     for i in uploaded_image:
    #         # try:
    #         with open(os.path.join('uploads_outwear',i.name),'wb') as f:
    #             f.write(i.getbuffer())
    #             return True
    #         # except:
    #     return False
    #     # 
        
    # def save_uploaded_image_pants(uploaded_image):
        
    #     for i in uploaded_image:
    #         # try:
    #         with open(os.path.join('uploads_pants',i.name),'wb') as f:
    #             f.write(i.getbuffer())
    #             return True
    #         # except:
    #     return False
        
        
    # def save_uploaded_image_shoes(uploaded_image):
        
    #     for i in uploaded_image:
    #         # try:
    #         with open(os.path.join('uploads_shoes',i.name),'wb') as f:
    #             f.write(i.getbuffer())
    #             return True
    #         # except:
    #     return False

    outwear = []
    pants = []
    shoes = []


    # uploaded_image = st.file_uploader('Upload outwear images',accept_multiple_files=True, key=1)

    # save_uploaded_image_outwear(uploaded_image)
            
    
            
            
    # uploaded_image = st.file_uploader('Upload pants images',accept_multiple_files=True, key=2)

    # save_uploaded_image_pants(uploaded_image)
            


    # uploaded_image = st.file_uploader('Upload shoes images',accept_multiple_files=True, key=3)

    # save_uploaded_image_shoes(uploaded_image)
            
            


    for file in os.listdir('uploads_outwear/'):
        
        outwear.append(os.path.join('uploads_outwear', file))
            
            
            
    for file in os.listdir('uploads_pants/'):
        
        pants.append(os.path.join('uploads_pants', file))
            
            
            
    for file in os.listdir('uploads_shoes/'):
        
        shoes.append(os.path.join('uploads_shoes', file))

            
            
    print(outwear)
    print(pants)
    print(shoes)

    values = []
    clothes = []

    # if(save_uploaded_image_outwear == True and save_uploaded_image_pants == True and save_uploaded_image_shoes == True):
    outwear_input = get_data_single(outwear[0])
    pants_input=get_data_single(pants[0])
    shoes_input = get_data_single(shoes[0])

    # outwear_input=get_data_single(outwear[0])
    # print(outwear_input)



    for i in range(3):
        for j in range(3):
            for k in range(3):
                if k==1:
                    outwear_input=get_data_single(outwear[i])
                    print(outwear_input)
                elif k==2 :
                    pants_input=get_data_single(pants[j])
                elif k==3 :
                    shoes_input=get_data_single(shoes[k])
                y_pred = model.predict([outwear_input,pants_input,shoes_input])
                values.append(y_pred[0][0])
                clothes.append([outwear[i], pants[j], shoes[k]])
                
                
                
    for value in range(len(values)):
        values[value] = round(values[value], 2)
        
        
    temp_2 = []


    for i in range(len(values)):
        temp_2.append((i,values[i]))
        
        
    from operator import itemgetter
    temp_2 = sorted(temp_2,key=itemgetter(1), reverse=True)


    seen = set()
    new = []
    for i in range(len(temp_2)):
        if temp_2[i][1] not in seen:
            seen.add(temp_2[i][1])
            new.append(temp_2[i])
            
            
    # col1, col2, col3  = st.columns([2,2,2])
    id_ = 0
    rows = []
    columns = []
    # if st.checkbox('Predict!'):
    for i in range(len(new)):
        index = new[i][0]
        count = 0
        columns = []
        for j in clothes[index]:
            print(j)
        # print(new[i][1])
    #         array = img.imread(j)
    #         plt.imshow(array)
        # print('-------------------------------------------------------------------------------------------------')
        # with st.expander('Show'):
            if count == 0:
                # with col1:
                    # array = cv2.imread(j)
                    # rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    # st.image(rgb)
                    columns.append(maping[j])
            if count == 1:
                # with col2:
                    # array = cv2.imread(j)
                    # rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    # st.image(rgb)
                    columns.append(maping[j])

            if count == 2:
                # with col3:
                    # array = cv2.imread(j)
                    # rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    # st.image(rgb)
                    columns.append(maping[j])
            count+=1
            id_+=1
            
        rows.append(columns)
  
    return jsonify({
    "status": "success",
    "prediction": rows
    # "confidence": str(classes[0][0][2]),
    # "upload_time": datetime.now()
})        # 