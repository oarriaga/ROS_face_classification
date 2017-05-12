#!/usr/bin/env python

PACKAGE = 'face_classification'
NODE = 'gender_classification'

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from face_classification.gender_classifier import GenderClassifier
from face_classification.utils import preprocess_image
from face_classification.face_detector import FaceDetector
from mcr_perception_msgs.msg import FaceList, Face
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
        self.package_path = roslib.packages.get_pkg_dir(PACKAGE)
        self.bridge = CvBridge()
        self.face_detector = FaceDetector(self.package_path +
                    '/trained_models/haarcascade_frontalface_default.xml')

        self.gender_classifier = GenderClassifier(self.package_path +
                    '/trained_models/gender_classifier.hdf5')
        self.face_list_publisher = rospy.Publisher('~faces', FaceList, queue_size=1)
        self.face_label_publisher = rospy.Publisher('~face_image', Image, queue_size=1)
        self.event_out_publisher = rospy.Publisher('~event_out', String, queue_size=1)
        self.event_in_subscriber = rospy.Subscriber('~event_in', String,
                                                    self.event_in_callback)
        self.save_image_path = self.package_path + '/images/gender_image.png'
        self.image = None
        self.event_in = None
        self.event_in_start_time = None
        self.x_offset = 20
        self.y_offset = 40

    def event_in_callback(self, msg):
        rospy.loginfo('EVENT_IN: {}'.format(msg.data))
        self.event_in_start_time = rospy.Time.now()
        self.event_in = msg
        #if self.event_in.data == 'e_trigger':
            #self.image_subscriber = rospy.Subscriber('~image', Image, self.image_callback)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as error:
            print(error)
        self.image = cv_image
        #self.image = cv2.resize(self.image, (360, 640))
        #self.image = cv2.flip(self.image, 0)
        #if self.image is not None:
        #    self.image = self.image[::-1,:]

    def adjust_rectangle(self, x, y, w, h, image_size):
        rospy.loginfo(image_size)
        if x < self.x_offset:
            x = self.x_offset
            pass
        if y < self.y_offset:
            y = self.y_offset
            pass
        if x + w + self.x_offset > image_size[0]:
            w = image_size[0] - 1 - x - self.x_offset
            pass
        if y + h + self.y_offset > image_size[1]:
            h = image_size[1] - 1 - y - self.y_offset
            pass
        return x, y, w, h

    def main_loop(self):
        if self.event_in:
            if self.event_in.data == 'e_trigger':
                self.event_out_publisher.publish(String('e_success'))
                self.image_subscriber = rospy.Subscriber('~image', Image, self.image_callback)
                if self.image is None:
                    rospy.logerr('NO IMAGE FROM CAMERA TOPIC')
                    return

                rospy.logerr('GOT IMAGE!!')
                self.image_subscriber.unregister()
                self.image = cv2.flip(self.image, 0)
                cv2.imwrite(self.package_path + '/window.jpg', self.image)
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                try:
                    faces = self.face_detector.detect(gray_image)
                except:
                    rospy.logerr(self.package_path)
                    self.image = None
                    return

                if len(faces) == 0:
                    rospy.logerr('NO FACES DETECTED')
                    self.image = None
                    return
                else:
                    face_objects = []
                    for (x, y, w, h) in faces:
                        x, y, w, h = self.adjust_rectangle(x, y, w, h, np.shape(self.image)[:2])
                        rospy.logerr(self.image.shape)
                        cv2.rectangle(self.image, (x - self.x_offset, y - self.y_offset),
                                      (x + w + self.x_offset, y + h + self.y_offset),
                                      (255, 0, 0), 2)
                        face = self.image[(y - self.y_offset):(y + h + self.y_offset),
                                          (x - self.x_offset):(x + w + self.x_offset)]
                        try:                        
                            face = cv2.resize(face, (48, 48))
                        except:
                            continue
                        face_obj = Face()
                        face_obj.image = self.bridge.cv2_to_imgmsg(face, 'bgr8')
                        face = np.expand_dims(face, 0)
                        face = preprocess_image(face)
                        predicted_label = self.gender_classifier.predict(face)
                        face_obj.gender = predicted_label
                        face_objects.append(face_obj)
                        cv2.putText(self.image, predicted_label, (x, y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0),
                                    1, cv2.CV_AA)
                        pass

                    face_list = FaceList()
                    face_list.faces = face_objects
                    self.face_list_publisher.publish(face_list)
                    output_image = Image()
                    output_image = self.bridge.cv2_to_imgmsg(self.image, 'bgr8')
                    self.face_label_publisher.publish(output_image)
                    #self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(self.save_image_path, self.image)
                    self.event_out_publisher.publish(String('e_success'))
                    self.event_in = None
                    self.image = None
            else:
                self.event_in = None

if __name__ == '__main__':
    rospy.init_node(NODE)
    rospy.loginfo("initializing node [%s]" % NODE)
    node = CNNGenderClassificationNode()
    while not rospy.is_shutdown():
        node.main_loop()
        rospy.sleep(0.5)
