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

import firebase_admin
from firebase_admin import credentials, storage


from firebase_admin import credentials, storage, initialize_app


service_account_key_path = 'serviceAccountKey.json'


cred = credentials.Certificate(service_account_key_path)
firebase_app = initialize_app(cred, {'storageBucket': 'fashionx-ebe6c.appspot.com'}, name='images_single')



def recommend_images_single(filename):
    
    def vgg():
        vgg16 = VGG16(weights='imagenet', input_shape = (224,224,3), include_top=False)
        for layers in vgg16.layers:
            layers.trainable=False
        
        return vgg16

    def features_image():
        features_images = np.array(pickle.load(open('embeddings_images_15000_recommend_vgg16.pkl', 'rb')))
        return features_images

    model = YOLO('best.pt')


    def extract(img_path, vgg16):
        print(img_path)
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
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
        
    # if os.path.exists('uploads/'):
    #     shutil.rmtree('uploads/')

    # else:
    os.makedirs('uploads/', exist_ok=True)


    ls = []

    index_pos_shirts = []
    index_pos_pants = []
    index_pos_shoes = []
    index_pos_shorts = []
    index_pos_jacket = []


    filenames = pickle.load(open('images_recommend_15000_filenames.pkl', 'rb'))


            

    # if os.path.exists('results'):
    #     shutil.rmtree('results')
        
    uid = uuid.uuid1()
    with open('uploads/{}.png'.format(uid), 'wb') as f:
        url = requests.get(filename)
        
        # body = url.body
        f.write(url.content)
    
    model.predict(os.path.join('uploads/', '{}.png'.format(uid)), save=True,  save_txt=True, save_crop=True, project='results')
        
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
  
    for file in os.listdir('results/predict/crops/'):
        similarity = []
        similarity_new = []
        for img in os.listdir('results/predict/crops/{}/'.format(file)):
            
            vgg16 = vgg()
            # gc.collect()
            features = extract('results/predict/crops/{}/'.format(file) + img, vgg16)
            
            for i in range(len(features_images)):

                similarity.append((cosine_similarity(features.reshape(1,-1), features_images[i].reshape(1, -1))))
            # print(similarity)
            similarity_new = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
            print(file)
    #             # indices=[]
    #             # ls=[]
    #             # print(file)
    #             similarity = []
    #             for img in os.listdir('results/predict/crops/{}/'.format(file)):
    #                 # print(img)

    #                 # image = Image.open('results/predict/crops/{}/'.format(file)+ img)
    #                 # image = preprocess('results/predict/crops/{}/'.format(file)+ img)
    #                 # print(img)

    #                 # features = resnet50.predict(image).flatten()
    #                 features=preprocess('results/predict/crops/{}/'.format(file)+ img)
    #                 # print(features)
    #                 # distances, indices = nn.kneighbors([features, n_neighbors=5])
    #                 # for i in range(len(features_images)):

    #                 # image = Image.open('results_hed/predict/crops/{}/'.format(file)+ img)
    #                 # print(img)

    #                 # features = vgg16.predict(image).flatten()
    #                 # print(features)
    #                 for i in range(len(features_images)):

    #                     similarity.append((cosine_similarity(features_images[i].reshape(1, -1), features.reshape(1,-1))))
    #                     # print(features)
    #                 # print(similarity)

    #                 # index_pos = 
    # #             # index_pos_'{}'.format(file) = []

    #                 # indices.append()
    #                     # print(features)
    #                 # print(similarity)
    #                 # for i in range(len(features_images)):

    #                     # similarity.append((cosine_similarity(features_images[i].reshape(1, -1), features.reshape(1,-1))))
    #                 # print(indices)
    #                 # print(distances)
    #             # print(indices[0][0])
    #                 # print(index_pos)
            # for i in range(5):
                # index_pos_'{}'.format(file) = []
                # print(file)
        if file == 'shirt':
            for i in range(10):
                # print(similarity[i][0])
                
                index_pos_shirts.append(similarity_new[i][0])
            # print(distances)

        elif file == 'shorts':
            for i in range(10):
                index_pos_shorts.append(similarity_new[i][0])
            # print(distances)

        elif file == 'pants':
            for i in range(10):
                index_pos_pants.append(similarity_new[i][0])
            # print(distances)


        elif file == 'shoe':
            for i in range(10):
                index_pos_shoes.append(similarity_new[i][0])
            # print(distances)
        elif file == 'jacket':
            for i in range(10):
                index_pos_jacket.append(similarity_new[i][0])

    print(index_pos_pants)
    print(index_pos_shirts)  
    print(index_pos_shoes)  
    print(index_pos_shorts)  
    print(index_pos_jacket)  

    
    filenames_pants = []
    filenames_shirts= []
    filenames_shoes = []
    filenames_shorts = []
    filenames_jacket = []
    # if st.checkbox('Show'):
    for file in os.listdir('results/predict/crops/'):
        if file == 'shorts':
     
                if len(index_pos_shorts) != 0:

                    # columns = st.columns(10)
                    for i in range(10):
                        # with columns[i]:
                            
                            temp = ' '.join(filenames[index_pos_shorts[i]].split('/')[-1:])
                            temp = temp.split('.')[-2]
                            # print(temp)
                            # print(filenames[similarity[0][0]].split('/')[-1:])
                            # path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                            # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                            # print('images/' + temp + '.jpg')
                            # image = cv2.imread(('images/' + temp + '.jpg'))
                            # image = cv2.resize(image, (224, 224))
                            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            # st.write(final_df.iloc[index_pos_shirts[i],4])
                            filenames_shorts.append('images/' + temp + '.jpg')
                #             st.image(image)

                # else:
                #     st.write('None')

        elif file == 'pants':
            # with st.expander('Top 5 recommndations for Pants'):
                if len(index_pos_pants) != 0:

                    # columns = st.columns(10)
                    for i in range(10):
                        # with columns[i]:
                        
                            print("Pants:", filenames[index_pos_pants[i]].split('/')[-1])
                            # print(filenames[index_pos_shorts[i]].split('/')[-1:])
                            # temp = ' '.join(filenames[index_pos_pants[i]].split('/')[:-1])
                            temp = filenames[index_pos_pants[i]].split('/')[-1]
                            # temp = temp.split('.')[-2]
                            # print(temp)
                            # print(filenames[similarity[0][0]].split('/')[-1:])
                            # path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                            # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                            # print('images/' + temp + '.jpg')
                            # image = cv2.imread(('images/' + temp + '.jpg'))
                            # image = cv2.resize(image, (224, 224))
                            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            # st.write(final_df.iloc[index_pos_shirts[i],4])
                            filenames_pants.append('images/' + temp)
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
                            # print(temp)
                            # print(filenames[similarity[0][0]].split('/')[-1:])
                            # path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                            # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                            # print('images/' + temp + '.jpg')
                            # image = cv2.imread(('images/' + temp + '.jpg'))
                            # image = cv2.resize(image, (224, 224))
                            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            # st.write(fina l_df.iloc[index_pos_shirts[i],4])
                            filenames_shirts.append('images/' + temp + '.jpg')
    #                         st.image(image)
    # #                                     # url = "https://www.streamlit.io"
    # #                                     # st.write("[Explore](%s)" % homepage_url)


    #             else:
    #                 st.write('None')

        elif file == 'shoe':
            # with st.expander('Top 5 recommndations for Shoes'):
                if len(index_pos_shoes) != 0:

                    # columns = st.columns(10)
                    for i in range(10):
                        # with columns[i]:
                            temp = ' '.join(filenames[index_pos_shoes[i]].split('/')[-1:])
                            temp = temp.split('.')[-2]
                            # print(temp)
                            # print(filenames[similarity[0][0]].split('/')[-1:])
                            # path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                            # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                            # print('images/' + temp + '.jpg')
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
                            # print(temp)
                            # print(filenames[similarity[0][0]].split('/')[-1:])
                            # path='/'.join(filenames[similarity[i][0]].split('/')[-1:])
                            # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                            # print('images/' + temp + '.jpg')
                            # image = cv2.imread(('images/' + temp + '.jpg'))
                            # image = cv2.resize(image, (224, 224))
                            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            # st.write(final_df.iloc[index_pos_shirts[i],4])
                            filenames_jacket.append('images/' + temp + '.jpg')
                #             st.image(ima  ge)

                # else:
                #     st.write('None')
        urls_shirts = []
        urls_shorts = []
        urls_pants = []
        urls_shoes = []
        urls_jacket = []
        
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
            
            
        
    return jsonify({
        "status": "success",
        "shirts" : urls_shirts,
        "shorts":urls_shorts,
        "shoes":urls_shoes,
        "pants":urls_pants,
        "jacket":urls_jacket,
        # "prediction": [filenames_shoes, filenames_pants, filenames_shirts, filenames_shorts, filenames_jacket],
        # "confidence": str(classes[0][0][2]),
        # "upload_time": datetime.now()
    })
