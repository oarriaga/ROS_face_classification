from __future__ import print_function
#! /usr/bin/env python2
#from scipy.misc import imresize
from keras.models import load_model
from numpy import argmax

class GenderClassifier:

    def __init__(self, model_path=None):

        self.labels = {0:'man',1:'woman'}
        if model_path != None:
            self.model_path = model_path
        else:
            self.model_path = '/trained_models/gender_classifier.hdf5'
        try:
            self.model = load_model(model_path)
        except:
            print('Failed to load model in:', model_path)

        self.image_size = (48, 48, 1)

    def predict(self, image_array):
        predictions = self.model.predict(image_array)
        predicted_arg = argmax(predictions)
        label = self.labels[predicted_arg]
        return label