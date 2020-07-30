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


def calculatepixelvalue(stats_load, img):
    row_num = stats_load.shape[1]
    diff = []
    for i, stat in enumerate(stats_load):
      centerx = stat[3]
      centery = stat[2]
      radius = stat[4]
      x0 = int(centerx-radius)   ##opencv image has its x as column, y as row, orign at left top
      y0 = int(centery-radius)   ##here we use x0 as row in array, y0 as column in array
      diameter = int(2*radius)
      pixelvalue = 0
      count = 0 #number of cilcle pixel
      for x in range(x0,x0+diameter+1):
        for y in range(y0,y0+diameter+1):
          if math.sqrt((x-centerx)**2 + (y-centery)**2)<=radius:
            pixelvalue = pixelvalue+img[x,y]
            count = count+1
      diff.append(stat[5]-pixelvalue)
    ####to do: fetch the two top max value and return

def callback(data,args):
    stats_load = args
    try:
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
    except CvBridgeError as e:
      print(e)
    
    ####calculate the two most pixelvalue changes

    
    # ## draw connected areas
    # for i, stat in enumerate(circle_connected):
    #   #绘制连通区域
    #   # cv.rectangle(cv_imageBGR, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (25, 25, 255), 3)
    #   cv.circle(cv_imageBGR,(stat[0], stat[1]), stat[2], (0,0,255))
    #   #按照连通区域的索引来打上标签
    #   cv.putText(cv_imageBGR, str(i+1), (stat[0], stat[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
    # cv.imshow("Image window", cv_imageBGR)
    
    # # # Draw detected blobs as red circles.
    # # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # # im_with_keypoints = cv.drawKeypoints(cv_imageBGR, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # # Show keypoints
    # # cv.namedWindow("Keypoints", cv.WINDOW_NORMAL)
    # # cv.imshow("Keypoints", im_with_keypoints)



    # # plt.subplot(111)
    # # plt.imshow(cv_image) 
    # # plt.xticks([])
    # # plt.yticks([])
    
    # cv.waitKey(3)


    
def image_subscriber():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    np.set_printoptions(suppress=True)
    print("listener starts")
    filename = '/home/ubuntu20/catkin_ws/src/PerceptionVisulization/logs/2020-07-29-15_13_05 detected circle stats.csv'
    data_load = np.loadtxt(filename,delimiter=",", skiprows=1)
    print("file name is ", filename)
    print("data read from the initialization file is:")
    print(data_load)

    rospy.init_node('image_subscriber', anonymous=True)
    rospy.Subscriber("image_raw", image_msg, callback, data_load)
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


