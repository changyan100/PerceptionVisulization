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

from std_msgs.msg import Float32MultiArray
import math


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
      diff_current = stat[5]-pixelvalue
      diff.append(diff_current)
      print("%dth fiber, init vaule %f, current value %d, diff %f" % (i, stat[5], pixelvalue, diff_current))
    ####to do: fetch the two top max value and return
    max1_index = diff.index(max(diff))  #return fiber index
    max1_value = max(diff)
    diff.remove(max(diff))
    max2_index = diff.index(max(diff))
    max2_value = max(diff)
    print("fibers responsed are: %d and %d, intensity changes are: %d and %d"\
          %(max1_index+1, max2_index+1, max1_value, max2_value))
    return [max1_index, max2_index, max1_value, max2_value]

def callback(data,args):
    stats_load = args
    # pub = args[1]
    bridge = CvBridge()
    try:
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
    except CvBridgeError as e:
      print(e)
    
    ####calculate the two most pixelvalue changes
    respondingfiber = calculatepixelvalue(stats_load, cv_image)
    index = respondingfiber[:2]
    
    ## draw the two responding circles
    cv_imageBGR = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR )
    for i, indexx in enumerate(index):
      x = stats_load[indexx, 2]
      y = stats_load[indexx, 3]
      r = stats_load[indexx, 4]
      #绘制连通区域
      cv.circle(cv_imageBGR,(int(x),int(y)), int(r), (0,0,255))
      #按照连通区域的索引来打上标签
      cv.putText(cv_imageBGR, str(indexx), (int(x), int(y) + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
    cv.imshow("Image window", cv_imageBGR)
    
    cv.waitKey(3)

    responding_arr = np.array(respondingfiber) 
    # pub.publish(responding_arr)

    


    
def image_process():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    np.set_printoptions(suppress=True)
    print("listener starts")
    file = '2020-07-30-16_04_19 detected circle stats.csv'
    filename = '/home/ubuntu20/catkin_ws/src/PerceptionVisulization/logs/' + file
    data_load = np.loadtxt(filename,delimiter=",", skiprows=1)
    print("file name is ", file)
    print("is the initialization file correct? [Y/N]")
    if(input()=='Y'or 'y'):

      print("data read from the initialization file is:")
      print(data_load)

      rospy.init_node('realtimeprocess_image_subscriber', anonymous=True)
      pub = rospy.Publisher('fiber_index', Float32MultiArray,queue_size=10)
      # rospy.Subscriber("image_raw", image_msg, callback, data_load, pub)
      rospy.Subscriber("image_raw", image_msg, callback, data_load)

      # spin() simply keeps python from exiting until this node is stopped
      rospy.spin()
    else:
      print("Program aborted! Please update filename in image_realtime_process_from_camera.py file")

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
    image_process()

