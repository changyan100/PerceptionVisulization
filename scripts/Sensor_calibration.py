#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sensor calibration code, read image from camera, detect fiber spots
calculate pixel value for each spot, ask user for loaded force,
save date (array include force, spot pixel vaule in sequence) to .csv
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


# cv_image 
# global stats_draw

low_th = 15  # filter too small spot by the area size  
high_th = 600 # filter too large connected area by the area size


class Calibration:

  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    # self.bridge = CvBridge()
    # self.image_sub = rospy.Subscriber("image_raw",Image,self.callback)
    np.set_printoptions(suppress=True)
    print("Calibration starts")
    fibernum = int(input("Please input fiber number: "))

    # self.datalen = int(input("Please input data length: "))
    datalen = 1000
    self.circle_dected = empty([datalen,3])
    self.datasave = empty([datalen,fibernum+1]) #[force, pixel1, pixel2,...]
    self.initflag = True
    self.count = 0

    # self.pub = rospy.Publisher("image_raw", image_msg, self.callback, fibernum)
    self.sub = rospy.Subscriber("image_raw", image_msg, self.callback, fibernum)


  def calculatepixelvalue(self, img, stats_load):
    row_num = stats_load.shape[0]
    pixel = []
    for i, stat in enumerate(stats_load):
      centerx = stat[1]
      centery = stat[0]
      radius = stat[2]
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
      pixel.append(pixelvalue)
   
    return pixel


  def ImageInit(self, cv_image):

    global low_th
    global high_th
    gray = cv_image
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
    #stats: a matrix with each line includes (x0, y0, width_x, hight_y, area)
    # here x0 means column, and y0 means rows, reverse to index of array!!!!!!!!!!!!!!!!!!
    # centroid: center_x, center_y
    ### remove small connected areas: diameter<5
    num = 0
    # circlenum = 24  #----------------------------defind circle number here
    # circle_dected = empty([circlenum,3])
    for i, stat in enumerate(stats):
      if stat[4]>low_th and stat[4]<high_th:
        # self.circle_dected[num,0] = int(stat[0] + stat[2]/2) # center_x
        # self.circle_dected[num,1] = int(stat[1] + stat[3]/2) # center_y
        self.circle_dected[num,0] = int(centroid[i,0]) # center_x
        self.circle_dected[num,1] = int(centroid[i,1]) # center_y
        r = int(stat[2]/2)+1 if stat[2]>stat[3] else int(stat[3]/2)+1 # radius
        self.circle_dected[num,2] = r+5 #enlarge detected circle radius
        num = num+1
    print("detected circle num = %d" %num)

    # global stats_draw

    # stats_draw = circle_dected

    return num

  def savedata_csv(self, filename, header, data):
    with open(filename,"w") as datacsv:
      csvwriter = csv.writer(datacsv,delimiter=',')
      csvwriter.writerow(header)
      csvwriter.writerows(data)

  def recordstats(self, stats):
    datalen = stats.shape[0]
    index = np.array([range(1,datalen+1)])
    indexarr = index.reshape([datalen,1])
    stats_cut = stats[:datalen,:]
    circle_stats = np.hstack((indexarr,stats)) #No, circleindex, center_x(column), center_y(row),radius,pixelvalue,pixelnum
    print("recorded stats:")
    print("index  force pixelvalue...")
    print(circle_stats)

    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    filename="/home/ubuntu20/catkin_ws/src/PerceptionVisulization/logs/"+now+r" calibration stats.csv"
    # filename = 'detected circle stats.csv'
    header = ["index","force(N)","pixelvalue..."]
    self.savedata_csv(filename, header, circle_stats)




  def callback(self, data,args):
    # global cv_image

    fibernum = args
    bridge = CvBridge()
    try:
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
    except CvBridgeError as e:
      print(e)
    
    if self.initflag:

      num = self.ImageInit(cv_image)
      # print(circle_dected)
      if not num == fibernum:
        print("detected fiber number does not match the input number!!! you may need modify detecting thresholds")
        print("current thresholds are:", high_th, low_th)
        ## draw the detected circles
        cv_imageBGR = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR )
        for i, indexx in enumerate(self.circle_dected):
          x = indexx[0]
          y = indexx[1]
          r = indexx[2]
        #绘制连通区域
          cv.circle(cv_imageBGR,(int(x),int(y)), int(r), (0,0,255))
        #按照连通区域的索引来打上标签
          cv.putText(cv_imageBGR, str(i+1), (int(x), int(y) + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
          if i>=fibernum-1:
            break
        cv.imshow("Image window", cv_imageBGR)
        print("press Esc to exit...")

        cv.waitKey()
        k = cv.waitKey(0)
        # k = cv2.waitKey(0) & 0xFF  # 64位机器
        if k == 27:         # 按下esc时，退出
        # if (input("press E to exit...")=='e'or'E'):
          cv.destroyAllWindows()
          rospy.signal_shutdown("Program aborted!")

      self.initflag = False
      print("Initilization finised!")
      ## draw the detected circles
      cv_imageBGR = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR )
      for i, indexx in enumerate(self.circle_dected):
        x = indexx[0]
        y = indexx[1]
        r = indexx[2]
      #绘制连通区域
        cv.circle(cv_imageBGR,(int(x),int(y)), int(r), (0,0,255))
      #按照连通区域的索引来打上标签
        cv.putText(cv_imageBGR, str(i+1), (int(x), int(y) + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
        if i>=fibernum-1:
          break
      cv.imshow("Image window", cv_imageBGR)
      print("press Esc to continue...")
      cv.waitKey()
      k = cv.waitKey(0)
      ## k = cv2.waitKey(0) & 0xFF  # 64位机器
      if k == 27:         # 按下esc时，退出
        cv.destroyAllWindows()



    else: #####init finished#############

      char = input("Please input loaded force or [E] to exit: ")

      if char=='e'or char =='E':
        # print(self.datasave)
        print("self count is: ", self.count)
        # clip zero row of datasave
        for i in range(self.datasave.shape[0]):
          valuesum = sum(self.datasave[i,:])
          if valuesum == 0:
            data2csv = self.datasave[:i,:]
            break
        print(data2csv)

        self.recordstats(data2csv)
        # self.recordstats(data2csv,self.count)
        rospy.signal_shutdown("Program aborted!")

          # If the command entered is a int, assign this value to rPRA
      try:
          force = float(char)
      except ValueError:
          rospy.signal_shutdown("Wrong input!")
    ####calculate pixelvalue for each spot
      pixels = self.calculatepixelvalue(cv_image,self.circle_dected)

      self.datasave[self.count,0] = force
      for i in range(fibernum):
        self.datasave[self.count,i+1] = pixels[i]

      self.count = self.count+1

      # print("Curret Image with force at %f" %force)
      # ## draw the detected circles
      # cv_imageBGR = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR )
      # for i, indexx in enumerate(self.circle_dected):
      #   x = indexx[0]
      #   y = indexx[1]
      #   r = indexx[2]
      # #绘制连通区域
      #   cv.circle(cv_imageBGR,(int(x),int(y)), int(r), (0,0,255))
      # #按照连通区域的索引来打上标签
      #   cv.putText(cv_imageBGR, str(i+1), (int(x), int(y) + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
      #   if i>=fibernum-1:
      #     break
      # cv.imshow("Image window", cv_imageBGR)
      # # print("press Esc to continue...")

      # cv.waitKey()
      # # k = cv.waitKey(0)
      # # ## k = cv2.waitKey(0) & 0xFF  # 64位机器
      # # if k == 27:         # 按下esc时，退出
      # #   cv.destroyAllWindows()

      # if (input("press E to exit...")=='e'or'E'):
      #   cv.destroyAllWindows()


def main(args):

  # try:
  #   signal.signal(signal.SIGINT, quit)
  #   signal.signal(signal.SIGTERM, quit)
  #   a = threading.Thread(target = imgshow)
  #   a.setDaemon(True)
  #   a.start()

  #   while True:
  #       pass
  # except(Exception,exc):
  #   print(exc)


  cal = Calibration()
  # rospy.init_node('image_converter', anonymous=True)
  rospy.init_node('calibration_image_subscriber', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)



