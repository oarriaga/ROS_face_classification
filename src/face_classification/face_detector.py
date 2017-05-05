from __future__ import print_function

import cv2

class FaceDetector(object):
    def __init__(self, model_path=None):

        if model_path != None:
            self.model_path = model_path
        else:
            self.model_path = '/trained_models/haarcascade_frontalface_default.xml'

        try:
            self.model = cv2.CascadeClassifier(self.model_path)
        except:
            print('Failed to load detection model in path:', self.model_path)

    def detect(self, gray_image):
        faces = self.model.detectMultiScale(gray_image, 1.3, 5)
        return faces
