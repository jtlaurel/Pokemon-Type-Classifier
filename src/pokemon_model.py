import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import io, color, filters

from PIL import Image
import requests
from io import BytesIO
from imageio import imread

class pokemonmodel(object):
    
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
       
    def get_file_paths(self, path):
        '''
        Takes a folder path and returns a set of all file paths of .jpg in the folders
        Input: Folder path
        '''
        file_set = set()

        for direct, _, files in os.walk(path):
            for file_name in files:
                rel_dir = os.path.relpath(direct, path)
                rel_file = os.path.join(rel_dir, file_name)
                if '.png' not in rel_file:
                    continue
                file_set.add(str(path)+rel_file)

        return file_set

    def one_hotify(self, y, n_classes=None):
        '''Convert array of integers to one-hot format;
        The new dimension is added at the end.'''
        if n_classes is None:
            n_classes = max(y) + 1
        labels = np.arange(n_classes)
        y = y[..., None]
        return (y == labels).astype(int)

    def load_images(self, path,size=(256,256)):

        file_paths = get_file_paths(path)

        images = []
        y = []
        for file in file_paths:
            img = keras.preprocessing.image.load_img(file, target_size=size)
            img_arr = keras.preprocessing.image.img_to_array(img.convert('RGBA'))[:,:,0:3]
            images.append(img_arr/255)
            y.append(file.split('/')[-1].split('_')[0])
        return images, y
    
    def read_data(self, path):
        self.df = pd.read_csv(path)
        
        self.pokedict = {}
        for i in range(df.shape[0]):
            if str(df['Secondary Type'].loc[i]) != 'nan':
                self.pokedict[df['Name'].loc[i]] = [df['Primary Type'].loc[i], df['Secondary Type'].loc[i]]
            else:
                self.pokedict[df['Name'].loc[i]] = [df['Primary Type'].loc[i]]
        return self.df
    
    def read_images(self,path):
        X,y_names = load_images((path))
        self.X = np.array(X)
        
        y_type = []
        for name in y_names:
            y_type.append(self.pokedict[name])

        y = pd.Series(y_type)
        self.y = pd.get_dummies(y.apply(pd.Series).stack()).sum(level=0)
        
        return self.X, self.y
    
    def split_data(self, size = 0.2, stratify = None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=46, test_size=size,stratify = stratify)
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, random_state=46, test_size=0.2,stratify = self.y_train)
        
        return self.X_train, self.X_val, self.X_test, self.y_Train, self.y_val, self.y_test
    
    def get_weights(self):
        self.class_dict = {}
        sums = np.sum(self.y, axis = 0)
        for i in range(y.shape[1]):
            self.class_dict[i] = sums[i]
            
        for key,value in class_dict.items():
            class_dict[key] = 1462/value
            
        return self.class_dict
                    
        
    def init_neural_net(self, model, window_size, pool_size, input_shape, threshold):
        self.model = model
        self.window_size = window_size
        self.pool_size = pool_size
        self.input_shape = input_shape
        self.threshold = threshold
        return self.model
     
    def create_neural_net(self, model):
        self.model.add(keras.layers.Conv2D(16, self.window_size, activation='relu', input_shape=self_input.shape))
        self.model.add(keras.layers.MaxPooling2D(pool_size=self.pool_size))
        self.model.add(keras.layers.Dropout(0.25))

        self.model.add(keras.layers.Conv2D(32, self.window_size, activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=self.pool_size))
        self.model.add(keras.layers.Dropout(0.25))

        self.model.add(keras.layers.Conv2D(64, self.window_size, activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=self.pool_size))
        self.model.add(keras.layers.Dropout(0.25))

        self.model.add(keras.layers.Conv2D(64, self.window_size, activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=self.pool_size))
        self.model.add(keras.layers.Dropout(0.25))

        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(18,
                        activation='sigmoid',
                        kernel_regularizer=keras.regularizers.l2(0.01)))

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(thresholds = self.threshold),tf.keras.metrics.Recall(thresholds = self.threshold)])
        
    def model_fit(self,epochs = 10, batch_size = 64):
        self.epoch = epochs
        self.batch_size = batch_size
        self.model.fit(self.X_train, self.y_train, epochs=self.epoch, validation_data=(self.X_val, self.y_val), batch_size=self.batch_size ,class_weight = self.class_dict, verbose = True)
        
    def model_evaluate(self, X, y):
        self.model.evaluate(X, y, verbose = True)
    
    def show_predictions(self, X):
        img = X
        classes = np.array(self.y_train.columns)
        proba = self.model.predict(img.reshape(1,256,256,3))
        top_3 = np.argsort(proba[0])[:-6:-1]
        for i in range(5):
            print("{}".format(classes[top_3[i]])+" {:.3}%".format(proba[0][top_3[i]]*100))
        io.imshow(img);
        
  