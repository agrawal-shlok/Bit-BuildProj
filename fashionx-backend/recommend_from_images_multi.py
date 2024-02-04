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

import uuid

from firebase_admin import credentials, storage, initialize_app


service_account_key_path = 'serviceAccountKey.json'


cred = credentials.Certificate(service_account_key_path)
firebase_app = initialize_app(cred, {'storageBucket': 'fashionx-ebe6c.appspot.com'}, name='images_multi')

def recommend_images_multi(filename):
    
    
    # if os.path.exists('uploads4/'):
    #     shutil.rmtree('uploads4/')

    # else:
    os.makedirs('uploads4/', exist_ok=True)
        
    def vgg():
        vgg16 = VGG16(weights='imagenet', input_shape = (224,224,3), include_top=False)
        for layers in vgg16.layers:
            layers.trainable=False
        
        return vgg16

    def features_image():
        features_images = np.array(pickle.load(open('embeddings_images_15000_recommend_vgg16.pkl', 'rb')))
        return features_images

    model = YOLO('best.pt')
    model_person = YOLO('yolov8s.pt')

    def extract(img_path, vgg16):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array,axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        
        result = vgg16.predict(preprocessed_img, verbose=0).flatten()
        return result

    def get_download_url(file_path):
   
            storage_ref = storage.bucket()
            print(file_path.split('/')[1])
            # Specify the file path within the storage bucket
            blob = storage_ref.blob(file_path.split('/')[1])
            print(blob)
            # Get the downloadable URL for the file
            download_url = blob.generate_signed_url(expiration=36000000000) # Set expiration time (in seconds)

            return download_url

   

    uid = uuid.uuid1()
    with open('uploads4/{}.png'.format(uid), 'wb') as f:
        url = requests.get(filename)
        
        # body = url.body
        f.write(url.content)

    
    def get_download_url(file_path):

        storage_ref = storage.bucket()
        print(file_path.split('/')[1])
        # Specify the file path within the storage bucket
        blob = storage_ref.blob(file_path.split('/')[1])
        print(blob)
        # Get the downloadable URL for the file
        download_url = blob.generate_signed_url(expiration=36000000000) # Set expiration time (in seconds)

        return download_url
        
    similarity_multi =[]
    def predict_multi(path, features_images):
        
        for file in os.listdir(path):
            similarity = []
            for img in os.listdir(os.path.join(path, file)):

                vgg16 = vgg()
                # gc.collect()
                features = extract(os.path.join(path, file,  img), vgg16)

                for i in range(len(features_images)):

                    similarity.append((cosine_similarity(features.reshape(1,-1), features_images[i].reshape(1, -1))))

                similarity_multi =  sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
            # print(file)
            if file == 'shirt':
                for i in range(10):
                    # print(similarity[i][0])
                    index_pos_shirts.append(similarity_multi[i][0])
                # print(distances)

            elif file == 'shorts':
                for i in range(10):
                    index_pos_shorts.append(similarity_multi[i][0])
                # print(distances)

            elif file == 'pants':
                for i in range(10):
                    index_pos_pants.append(similarity_multi[i][0])
                # print(distances)


            elif file == 'shoe':
                for i in range(10):
                    index_pos_shoes.append(similarity_multi[i][0])
                # print(distances)
                
            elif file == 'jacket':
                for i in range(10):
                    index_pos_jacket.append(similarity_multi[i][0])
                        
        return similarity_multi

    
    
    ls = []

    index_pos_shirts = []
    index_pos_pants = []
    index_pos_shoes = []
    index_pos_shorts = []
    index_pos_jacket = []


    filenames = pickle.load(open('images_recommend_15000_filenames.pkl', 'rb'))


            

    # if st.checkbox('Upload a multi-person photo: '):
    # uploaded_image = st.file_uploader('Upload a group image')
    
    # if os.path.exists('uploads4/'):
    #     shutil.rmtree('uploads4/')
    #     os.mkdir('uploads4/')
    # # else:
    # else:
    #     os.mkdir('uploads4/')
        
    # if uploaded_image is not None:
        
    #     if save_uploaded_image_multi(uploaded_image):
            
            # if os.path.exists('results'):
            #     shutil.rmtree('results')
            # else:
            #     os.mkdir('results')

    # if os.path.exists('results_multi'):
    #     shutil.rmtree('results_multi')
    
    # if os.path.exists('results_persons'):
    #     shutil.rmtree('results_persons')
    # if os.path.exists('results_groups'):
    #     shutil.rmtree('results_groups')
    
    
    model_person.predict(os.path.join('uploads4/{}.png'.format(uid)), save=True,  save_txt=True, save_crop=True, project='results_persons')  
    # gc.collect()

#                 for file in os.listdir('results/'):

#                     features = extract('results/' + file)

            
            # print(similarity)

            # for i in range(5):
            #     st.image('myntradataset/images/' + similarity)



#                 for file in os.listdir('results/predict/crops/'):

#                     if file in ['shirt', 'jacket', 'dress']:

#                         file2 = 'shirt'

#                         os.rename('results/predict/crops/{}/'.format(file), 'results/predict/crops/{}/'.format(file2))
    features_images = features_image()
    number = 0
    json_obj = {}
    # if st.checkbox('Recommend'):
    for person in os.listdir('results_persons/predict/crops/person/'):
        ls = []

        index_pos_shirts = []
        index_pos_pants = []
        index_pos_shoes = []
        index_pos_shorts = []
        index_pos_jacket = []
        
        similarity_multi = []
        # similarity = []
        model.predict(os.path.join('results_persons/predict/crops/person/{}'.format(person)), save=True,  save_txt=True, save_crop=True, project='results_multi')  
        # for img in os.listdir('results_multi/predict/crops/persons/'.format(file)):
            
        predict_multi('results_multi/predict/crops/', features_images)
      
          
        # print(index_pos_pants)
        # print(index_pos_shirts)  
        # print(index_pos_shoes)  
        # print(index_pos_shorts)  
        # print(index_pos_jacket)  

        filenames_pants = []
        filenames_shirts= []
        filenames_shoes = []
        filenames_shorts = []
        filenames_jacket = []
    # if st.checkbox('Show'):
    # with st.expander('Person_{}'.format(number)):
        # st.image('results_multi/predict/crops/persons/'.format(file) + img)
        # print("HIUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        for pred in os.listdir('results_multi/'):
            # print("HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            for file in os.listdir('results_multi/' + pred + '/' 'crops/'):
                print('results_multi/' + pred)
                if file == 'shorts':
                    # with st.expander('Top 5 recommndations for short'):
                        if len(index_pos_shorts) != 0:

                            # columns = st.columns(10)
                            for i in range(10):
                                # with columns[i]:
                                    temp = ' '.join(filenames[index_pos_shorts[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    # path='/'.join(filenames[similarity_multi[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    # image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])

                                    # st.image(image)
                                    filenames_shorts.append('images/' + temp + '.jpg')

                        # else:
                        #     st.write('None')

                elif file == 'pants':
                    # with st.expander('Top 5 recommndations for Pants'):
                        if len(index_pos_pants) != 0:

                            # columns = st.columns(10)
                            for i in range(10):
                                # with columns[i]:
                                    temp = ' '.join(filenames[index_pos_pants[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    # path='/'.join(filenames[similarity_multi[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    # image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])
                                    filenames_pants.append('images/' + temp + '.jpg')
                        #             st.image(image)

                        # else:
                        #     st.write('None')

                elif file == 'shirt':
                    # with st.expander('Top 5 recommndations for Shirts'):
                        if len(index_pos_shirts) != 0:


                            # columns = st.columns(10)
                            for i in range(10):
                                # with columns[i]:
                                    # homepage_url = final_df.iloc[index_pos_shirts[i],2]
                                        # print(filenames[similarity[i][0]])
                                    temp = ' '.join(filenames[index_pos_shirts[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    # path='/'.join(filenames[similarity_multi[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    # image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])
                                    filenames_shirts.append('images/' + temp + '.jpg')
        #                             st.image(image)
        # #                                     # url = "https://www.streamlit.io"
        # #                                     # st.write("[Explore](%s)" % homepage_url)


        #                 else:
        #                     st.write('None')

                elif file == 'shoe':
                    # with st.expander('Top 5 recommndations for Shoes'):
                        if len(index_pos_shoes) != 0:

                            # columns = st.columns(10)
                            for i in range(10):
                                # with columns[i]:
                                    temp = ' '.join(filenames[index_pos_shoes[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    path='/'.join(filenames[similarity_multi[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    # image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])
                                    filenames_shoes.append('images/' + temp + '.jpg')
                        #             st.image(image)

                        # else:
                        #     st.write('None')
                elif file == 'jacket':
                    # with st.expander('Top 5 recommndations for Jacket'):
                        if len(index_pos_jacket) != 0:

                            # columns = st.columns(10)
                            for i in range(10):
                                # with columns[i]:
                                    temp = ' '.join(filenames[index_pos_jacket[i]].split('/')[-1:])
                                    temp = temp.split('.')[-2]
                                    print(temp)
                                    # print(filenames[similarity[0][0]].split('/')[-1:])
                                    # path='/'.join(filenames[similarity_multi[i][0]].split('/')[-1:])
                                    # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                                    print('images/' + temp + '.jpg')
                                    # image = cv2.imread(('images/' + temp + '.jpg'))
                                    # image = cv2.resize(image, (224, 224))
                                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # st.write(final_df.iloc[index_pos_shirts[i],4])
                                    filenames_jacket.apend('images/' + temp + '.jpg')
    
    
    
   #             st.image(image)
        urls_shirts = []
        urls_shorts = []
        urls_pants = []
        urls_shoes = []
        urls_jacket = []
        print(filenames_shorts)
        print(filenames_pants)
        print(filenames_shirts)
        for i in filenames_shirts:
            url = get_download_url(i)
            urls_shirts.append(url)
        
        for i in filenames_shoes:
            url = get_download_url(i)
            urls_shoes.append(url)
            
        for i in filenames_shorts:
            url = get_download_url(i)
            urls_shorts.append(url)
            
        for i in filenames_pants:
            url = get_download_url(i)
            urls_pants.append(url)
            
        for i in filenames_jacket:
            url = get_download_url(i)
            urls_jacket.append(url)
            
            
                # else:
        json_obj[number]  = {
        "status": "success",
        "shirts" : urls_shirts,
        "shorts":urls_shorts,
        "shoes":urls_shoes,
        "pants":urls_pants,
        "jacket":urls_jacket,
        }
        number+=1
    return json_obj


