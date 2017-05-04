#!/usr/bin/env python

PACKAGE = 'face_classification'
NODE = 'emotion_classification'

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
#from face_classification.emotion_classifier import EmotionClassifier
from face_classification.gender_classifier import GenderClassifier
from face_classification.utils import preprocess_image
from face_classification.face_detector import FaceDetector
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import roslib

class CNNGenderClassificationNode:
    """
    ROS wrapper for face_classification model
    """
    IMAGE_CACHE_SIZE = 30

    def __init__(self):

        rospy.init_node(NODE, log_level=rospy.DEBUG)
        self.package_path = roslib.packages.get_pkg_dir(PACKAGE)
        self.bridge = CvBridge()
        self.face_detector = FaceDetector(self.package_path +
                    '/trained_models/haarcascade_frontalface_default.xml')

        self.emotion_classifier = GenderClassifier(self.package_path +
                    '/trained_models/gender_classifier.hdf5')
        self.event_out_publisher = rospy.Publisher('~event_out', String)
        self.event_in_subscriber = rospy.Subscriber('~event_in', String,
                                                self.event_in_callback)
        self.save_image_path = self.package_path + '/images/gender_image.png'
        self.image = None
        self.event_in = None
        self.event_in_start_time = None
        self.x_offset = 30
        self.y_offset = 60

    def event_in_callback(self, msg):
        rospy.loginfo('EVENT_IN: {}'.format(msg.data))
        self.event_in_start_time = rospy.Time.now()
        self.event_in = msg

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as error:
            print(error)
        self.image = cv_image

    def main_loop(self):
        if self.event_in:
            if self.event_in.data == 'e_trigger':
                self.event_out_publisher.publish(String('e_success'))
                self.image_subscriber = rospy.Subscriber('/camera/rgb/image_raw',
                                                Image, self.image_callback)
                if self.image == None:
                    rospy.logerr('NO IMAGE FROM CAMERA TOPIC')
                    return
                self.image_subscriber.unregister()
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                try:
                    faces = self.face_detector.detect(gray_image)
                except:
                    rospy.logerr(self.package_path)
                    return

                if len(faces) == 0:
                    rospy.logerr('NO FACES DETECTED')
                    return
                else:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(self.image, (x - self.x_offset, y - self.y_offset),
                                        (x + w + self.x_offset, y + h + self.y_offset),
                                                                        (255, 0, 0), 2)
                        face = self.image[(y - self.y_offset):(y + h + self.y_offset),
                                          (x - self.x_offset):(x + w + self.x_offset)]
                        face = cv2.resize(face, (48, 48))
                        face = np.expand_dims(face, 0)
                        face = preprocess_image(face)
                        predicted_label = self.emotion_classifier.predict(face)
                        cv2.putText(self.image, predicted_label, (x, y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0),
                                                                    1, cv2.LINE_AA)
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(self.save_image_path, self.image)
                    self.event_out_publisher.publish(String('e_success'))
                    self.event_in = None

if __name__ == '__main__':
    node = CNNGenderClassificationNode()
    while not rospy.is_shutdown():
        node.main_loop()
        rospy.sleep(0.1)