from __future__ import print_function
#! /usr/bin/env python2
#from scipy.misc import imresize
from keras.models import load_model
from numpy import argmax

class EmotionClassifier:

    def __init__(self, model_path=None):

        self.labels = {0:'angry',1:'disgust',2:'sad',3:'happy',
                                4:'sad',5:'surprise',6:'neutral'}

        if model_path != None:
            self.model_path = model_path
        else:
            self.model_path = '/trained_models/emotion_classifier_v2.hdf5'

        try:
            self.model = load_model(model_path)
        except:
            print('Failed to load model in:', model_path)

        self.image_size = (48, 48, 1)


    def predict(self, image_array):
        #image_array = imresize(image_array, size=self.image_size[:2])
        #image_array = image_array / 255.0
        #image_array = expand_dims(image_array,0)
        predictions = self.model.predict(image_array)
        predicted_arg = argmax(predictions)
        label = self.labels[predicted_arg]
        return label
