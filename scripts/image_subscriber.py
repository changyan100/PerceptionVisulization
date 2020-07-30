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



def callback(data):
    Datamatrix = data.data
    bridge = CvBridge()
    # rospy.loginfo("I heard %0.6f", data.data)
    try:
      # cv_image = bridge.imgmsg_to_cv2(data, "mono8")
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
      gray = cv_image

      ##### initilization ##################excute only once in the beginning#################33
      ##### -binarization 
      ##### -Hough circle, 
      ##### - find max diameter

      # binarization 

      # Global threshoding
      # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE) 
      # # print("threshold value: %s"%ret)
      # partitial thresholding
      # binary1 =  cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10) 
      ##### max gradient threshold method
      h, w =cv_image.shape[:2]
      m = np.reshape(gray, [1,w*h])  #<size> (1, 316224) #turple 元祖类型
      m_sort = sorted(m[0,:], reverse=True)
      step = 30
      v2 = m_sort[:-step]
      v1 = m_sort[step:]
      m_sort_diff = []
      for i in range(0,(len(m_sort)-step-1)):
      	diff = v2[i]-v1[i]
      	m_sort_diff.append(diff)

      max_index = m_sort_diff.index(max(m_sort_diff))
      # print("max index = %d" %max_index)
      thresh = m_sort[max_index+step]
      # print("tresh = %d" %thresh)
      ret, binary =  cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)

      # canny = cv.Canny(binary, 50, 150)  # 50是最小阈值,150是最大阈值

      ####### hough circle
      # circle = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=50, maxRadius=300)
      circle = cv.HoughCircles(binary, cv.HOUGH_GRADIENT,1.4, 1,param1=300, param2=21,minRadius=5,maxRadius=20)
      									#dpi, minDist 
      # circle = cv.HoughCircles(cv_image, cv.HOUGH_GRADIENT,1, 1,param1=255, param2=25,minRadius=5,maxRadius=20)
      									#dpi, minDist 

      if not circle is None:
        circle = np.uint16(np.around(circle))
        print(circle)
      else:
      	print("no circle is detected!")


      #### detect connected areas and their centers
      # 搜索图像中的连通区域
      ret, labels, stats, centroid = cv.connectedComponentsWithStats(binary)


      # # Set up the detector with default parameters.
      # detector = cv.SimpleBlobDetector()
      # # Detect blobs.
      # keypoints = detector.detect(binary)

    except CvBridgeError as e:
      print(e)

    cv_imageBGR = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR ) 

    cv.namedWindow("image_raw", cv.WINDOW_NORMAL)
    cv.imshow("image_raw", cv_image)  
    # cv.namedWindow("binary", cv.WINDOW_NORMAL)
    # cv.imshow("binary", binary)
    
    # cv.circle(cv_image, (50,50), 10, 255)  #draw a circle for each light spot
   

    ### draw hough circles 
    # if not circle is None:
    #   count = 0
    #   for i in circle[0, :]:
    #     cv.circle(cv_imageBGR, (i[0], i[1]), i[2], (0, 0, 255))
    #     count = count+1
    #   print("number of circle = %d" %count )
    # cv.imshow("Image window", cv_imageBGR)
    # cv.waitKey(3)

    ### remove small connected areas: diameter<5
    num = 0
    circle_connected = empty([24,3])
    for i, stat in enumerate(stats):
      if stat[4]>100:
        circle_connected[num,0] = stat[0] + stat[2]/2
        circle_connected[num,1] = stat[1] + stat[3]/2
        circle_connected[num,2] = stat[2] if stat[2]>stat[3] else stat[3]
        num = num+1
    print("num = %d" %num)

    ## draw connected areas
    for i, stat in enumerate(circle_connected):
      #绘制连通区域
      # cv.rectangle(cv_imageBGR, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (25, 25, 255), 3)
      cv.circle(cv_imageBGR,(stat[0], stat[1]), stat[2], (0,0,255))
      #按照连通区域的索引来打上标签
      cv.putText(cv_imageBGR, str(i+1), (stat[0], stat[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
    cv.imshow("Image window", cv_imageBGR)
    
    # # Draw detected blobs as red circles.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv.drawKeypoints(cv_imageBGR, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # Show keypoints
    # cv.namedWindow("Keypoints", cv.WINDOW_NORMAL)
    # cv.imshow("Keypoints", im_with_keypoints)



    # plt.subplot(111)
    # plt.imshow(cv_image) 
    # plt.xticks([])
    # plt.yticks([])
    
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


