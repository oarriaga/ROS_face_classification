cmake_minimum_required(VERSION 2.8.3)
project(face_classification)

find_package(catkin REQUIRED COMPONENTS
  mcr_perception_msgs
  roslint
  rospy
  std_msgs
)

find_package(OpenCV REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS
    mcr_perception_msgs
    std_msgs
)

catkin_python_setup()

roslint_python()

# INSTALL
install(PROGRAMS
  nodes/emotion_classification.py
  nodes/gender_classification.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
