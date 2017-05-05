#!/usr/bin/env python

PACKAGE = 'face_classification'
NODE = 'emotion_classification'

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from face_classification.emotion_classifier import EmotionClassifier
from face_classification.utils import preprocess_image
from face_classification.face_detector import FaceDetector
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import roslib

class CNNEmotionClassificationNode:
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

        self.emotion_classifier = EmotionClassifier(self.package_path +
                    '/trained_models/emotion_classifier_v2.hdf5')
        self.event_out_publisher = rospy.Publisher('~event_out', String, queue_size=1)
        self.event_in_subscriber = rospy.Subscriber('~event_in', String,
                                                self.event_in_callback)
        self.save_image_path = self.package_path + '/images/image.png'
        self.image = None
        self.event_in = None
        self.event_in_start_time = None

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

    def main_loop2(self):
        if self.event_in:
            if self.event_in.data == 'e_trigger':
                self.bridge.imgmsg_to_cv2()
                """
                if self.object_list:
                    self.process_object_list()
                    self.result_publiser.publish(self.object_list)
                    self.object_list = None
                    self.event_in = None
                    self.event_out_publisher.publish(String('e_success'))
                else:
                    delay = rospy.Time.now() - self.event_in_start_time
                    if delay<rospy.Duration(10.0):
                        rospy.logwarn('WAITING FOR OBJECT LIST FOR {}/10 SEC'.format(delay.to_sec()))
                        return
                    self.event_in = None
                    rospy.logerr('NO OBJECT LIST RECIEVED')
                    self.event_out_publisher.publish(String('e_failure'))
                """
    def main_loop(self):
        if self.event_in:
            if self.event_in.data == 'e_trigger':
                self.event_out_publisher.publish(String('e_success'))
                self.image_subscriber = rospy.Subscriber('~image', Image, self.image_callback)
                if self.image == None:
                    rospy.logerr('NO IMAGE FROM CAMERA TOPIC')
                    return
                self.image_subscriber.unregister()
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
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
                        cv2.rectangle(gray_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        face = gray_image[y:(y + h),x:(x + w)]
                        face = cv2.resize(face, (48, 48))
                        face = np.expand_dims(face, 0)
                        face = np.expand_dims(face, -1)
                        face = preprocess_image(face)
                        predicted_label = self.emotion_classifier.predict(face)
                        cv2.putText(gray_image, predicted_label, (x, y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0),
                                    1, cv2.CV_AA)
                    cv2.imwrite(self.save_image_path, gray_image)
                    self.event_out_publisher.publish(String('e_success'))
                    self.event_in = None

if __name__ == '__main__':
    node = CNNEmotionClassificationNode()
    while not rospy.is_shutdown():
        node.main_loop()
        rospy.sleep(0.1)
