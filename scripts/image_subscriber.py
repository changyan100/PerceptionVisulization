#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
subscribe image from camera and try to do initializaiton in callback()
didnot finished, so the code is not runable
refe Initializaiton code in Initializaiton_circledetected.py file
'''

import rospy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from std_msgs.msg import Float64

import threading, time, signal
import sys
# from PIL import Image
# import PIL


from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as image_msg
import cv2 as cv

from numpy import empty



# class imageshow:

#      def __init__(self):
#         self.image

def callback(data):

    # Datamatrix = data.data

    bridge = CvBridge()
    # rospy.loginfo("I heard %0.6f", data.data)
    try:
      # cv_image = bridge.imgmsg_to_cv2(data, "mono8")
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
      gray = cv_image


   
    except CvBridgeError as e:
      print(e)


    cv_imageBGR = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR ) 


    cv.namedWindow("image_raw", cv.WINDOW_NORMAL)
    cv.imshow("image_raw", cv_image)  
    # cv.namedWindow("binary", cv.WINDOW_NORMAL)
  
    
    cv.waitKey(3)


    
def image_subscriber():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    print("listener starts")
    rospy.init_node('image_subscriber', anonymous=True)
    rospy.Subscriber("image_raw", image_msg, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    # listener()
    # try:
    #     signal.signal(signal.SIGINT, quit)
    #     signal.signal(signal.SIGTERM, quit)
    #     a = threading.Thread(target = draw)
    #     # b = threading.Thread(target = printB)
    #     a.setDaemon(True)
    #     a.start()
    #     image_subscriber()
    #     # b.setDaemon(True)
    #     # b.start()

    #     while True:
    #         pass
    # except(Exception,exc):
    #     print(exc)
    image_subscriber()


