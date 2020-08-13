#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Initialization code, well-organized in a class
fetch image from camera
detect circles, calculate pixels
reorder the circle index with user's input
save date to .csv and save img as .png
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
import math
import csv

import os,sys
import csv
import time


# globalflag = 0
# circlenum = 4

low_th = 10  # filter too small spot by the area size  
high_th = 600 # filter too large connected area by the area size

######## !! to do: coding to search for good threshold to detect exact circle number

class Initilization:

  def __init__(self, cv_image, circlenum):

    global globalflag
    global low_th  # filter too small spot by the area size
    global high_th # filter too large connected area by the area size

    print("--------------Initilization starts............")
    np.set_printoptions(suppress=True)
    # cv_image = cv.imread("image_raw_fibers.png")
        # test 
    # cv_image = self.drawcicle(cv_image)
    gray = cv_image
    cv_imageBGR = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR ) 
    # gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY ) 
    # cv_imageBGR =  cv_image # 

    ##### max gradient threshold method
    h, w =gray.shape[:2]
    # print(gray.shape)
    # print("h=%d" %h)
    # print("w=%d" %w)
    # print("w*h=%d" %(w*h))
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


    #### detect connected areas and their centers
    # 搜索图像中的连通区域
    ret, labels, stats, centroid = cv.connectedComponentsWithStats(binary)
    #ret: number of labels
    #labels: label each connected area from 1 to n
    #stats: a matrix with each line includes (center_x, center_y, width_x, hight_y, area)
    # here center_x means column, and center_y means rows, reverse to index of array!!!!!!!!!!!!!!!!!!

    ### remove small connected areas: diameter<5
    num = 0
    # circlenum = 24  #----------------------------defind circle number here
    circle_dected = empty([circlenum,3])
    for i, stat in enumerate(stats):
      if stat[4]>low_th and stat[4]<high_th:
        circle_dected[num,0] = int(stat[0] + stat[2]/2) # center_x
        circle_dected[num,1] = int(stat[1] + stat[3]/2) # center_y
        r = int(stat[2]/2)+1 if stat[2]>stat[3] else int(stat[3]/2)+1 # radius
        circle_dected[num,2] = r + 5  #-----------------manually enlarge the circle by 5 pixel------------------
        num = num+1
    print("detected circle num = %d" %num)
    # print(circle_dected)

    pixelvalue = []
    pixelnum = []
    circle_pixel = empty([circlenum,2])
    #### calculate total pixel value in each circle
    for i, stat in enumerate(circle_dected):
      pixelvalreturn, pixelnumreturn = self.calculatepixelvalue(stat[0], stat[1], stat[2], gray) 
      pixelvalue.append(pixelvalreturn)
      pixelnum.append(pixelnumreturn)
      circle_pixel[i,0] = pixelvalreturn
      circle_pixel[i,1] = pixelnumreturn
      # print("%dth circle pixelvalue is: %d, pixelnum is %d" %(i+1,pixelvalreturn, pixelnumreturn))
    
    index = np.array([range(1,circlenum+1)])
    indexarr = index.reshape([circlenum,1])

    circle_stats = np.hstack((indexarr,circle_dected, circle_pixel)) #index, center_x(column), center_y(row),radius,pixelvalue,pixelnum

    # np.savetxt('circle_stats.csv',circle_stats,delimiter=',')

    # cv.namedWindow("image_raw", cv.WINDOW_NORMAL)
    # cv.imshow("image_raw", cv_image)  

    ## draw connected areas
    for i, stat in enumerate(circle_dected):
      #绘制连通区域
      # cv.rectangle(cv_imageBGR, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (25, 25, 255), 3)
      cv.circle(cv_imageBGR,(int(stat[0]), int(stat[1])), int(stat[2]), (0,0,255))
      #按照连通区域的索引来打上标签
      cv.putText(cv_imageBGR, str(i+1), (int(stat[0]), int(stat[1])+ 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 25, 25),1)
    # cv.circle(cv_imageBGR,(10,50),2,(0,0,255))
    cv.namedWindow("Image window", cv.WINDOW_NORMAL)
    cv.imshow("Image window", cv_imageBGR)
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    picname="/home/ubuntu20/catkin_ws/src/PerceptionVisulization/logs/"+now+r" detected circle stats.png"
    cv.imwrite(picname, cv_imageBGR)

    print("please take note of the circle index for each fiber, later on you need to type in thoes indexs")
    print("press Esc to continue...")

        # cv.waitKey()
    k = cv.waitKey(0)
    ## k = cv2.waitKey(0) & 0xFF  # 64位机器
    if k == 27:         # 按下esc时，退出
      cv.destroyAllWindows()


    self.reorderstats(circle_stats,circlenum)

    print("----Initilization finished succesfully, data is saved to log folder!")
    globalflag = 1



  def calculatepixelvalue(self, center_x, center_y, radius, img):
    centerx = center_y
    centery = center_x
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
    return pixelvalue, count


  def drawcicle(self, img):
    orign_x = 100
    orign_y = 100
    radius = 10
    x0 = orign_x-radius
    y0 = orign_y-radius
    count = 0
    for x in range(x0,orign_x+radius*2+1):
      for y in range(y0,orign_y+radius*2+1):
        if math.sqrt((x-orign_x)**2 + (y-orign_y)**2)<=radius:
          img[x,y,0]=255
          img[x,y,1]=255
          img[x,y,2]=255
          count = count+255
    print("drawd circle pixelvalue is %d" %count)
    return img

  def savedata_csv(self, filename, header, data):
    with open(filename,"w") as datacsv:
      csvwriter = csv.writer(datacsv,delimiter=',')
      csvwriter.writerow(header)
      csvwriter.writerows(data)


  def reorderstats(self, stats,circlenum):
    stats_reordered = empty([circlenum,6])
    for i, stat in enumerate(stats):
      print("Please input the circle index for fiber %d" %(i+1))
      circleindex = int(input())-1
      stats_reordered[i,:] = stats[circleindex,:]


    index = np.array([range(1,circlenum+1)])
    indexarr = index.reshape([circlenum,1])

    circle_stats = np.hstack((indexarr,stats_reordered)) #No, circleindex, center_x(column), center_y(row),radius,pixelvalue,pixelnum
    print("detected circle stats:")
    print("fiberNo  circleindex centerx centery radius pixelnum pixelnum")
    print(circle_stats)

    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    filename="/home/ubuntu20/catkin_ws/src/PerceptionVisulization/logs/"+now+r" detected circle stats.csv"
    # filename = 'detected circle stats.csv'
    header = ["fiberNo","circleindex","center_x(column)","center_y(row)","radius","pixelvalue","pixelnum"]
    self.savedata_csv(filename, header, circle_stats)


def callback(data):

    # global globalflag
    # global circlenum
    bridge = CvBridge()

    try:
        # cv_image = bridge.imgmsg_to_cv2(data, "mono8")
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
        # cv_image = cv.imread("image_raw_fibers.png")
    except CvBridgeError as e:
      print(e)
    
    a = input("press [I] to show image, press [C] to start initialization, press [E] to exit program: ")
    
    if a == 'I' or a == 'i':
      cv.namedWindow("image_raw", cv.WINDOW_NORMAL)
      cv.imshow("image_raw", cv_image) 
      print("press Esc to continue...")
        # cv.waitKey()
      k = cv.waitKey(0)
    ## k = cv2.waitKey(0) & 0xFF  # 64位机器
      if k == 27:         # 按下esc时，退出
        cv.destroyAllWindows()
    elif a == 'C' or a == 'c':
      
      circlenum = int(input("please input fiber number:"))

      Init = Initilization(cv_image,circlenum)
    elif a == 'E' or a == 'e':
      print("Program is aborted by the user!")
      rospy.signal_shutdown("Program aborted!")
    else: 
      print("Wrong input!")


    # if globalflag ==0:
    #   # circlenum = 25   ###########update circle num here!!!!!!!!
     
    #   # rospy.loginfo("I heard %0.6f", data.data)
    #   Init = Initilization(cv_image,circlenum)

    # else:


    #   print("press Ctrl+C to end initialization")


 
def image_subscriber():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    print("Initilization listener starts")
    rospy.init_node('Initialization_image_subscriber', anonymous=True)
    rospy.Subscriber("image_raw", image_msg, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()    



if __name__ == '__main__':
  # cv_image = cv.imread("image_raw_fibers.png")
  # circlenum = 24
  # Init = Initilization(cv_image,circlenum)
  image_subscriber()
