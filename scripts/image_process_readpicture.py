#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Initialization code, use static img, codes are not well-organized, which are seperated in functions
but.... this file can run successfully..
this code does circle detection, pixel calculation, and save img and stats, but does not reorder the stats
for complete initialization code, see Initializaiton_circledetected.py file
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
    cv.imwrite('image_raw_fibers.png',cv_image)
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
    # num = 0
    # circle_connected = empty([24,3])
    # for i, stat in enumerate(stats):
    #   if stat[4]>100:
    #     circle_connected[num,0] = stat[0] + stat[2]/2
    #     circle_connected[num,1] = stat[1] + stat[3]/2
    #     circle_connected[num,2] = stat[2] if stat[2]>stat[3] else stat[3]
    #     num = num+1
    # print("num = %d" %num)

    # ## draw connected areas
    # for i, stat in enumerate(circle_connected):
    #   #绘制连通区域
    #   # cv.rectangle(cv_imageBGR, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (25, 25, 255), 3)
    #   cv.circle(cv_imageBGR,(stat[0], stat[1]), stat[2], (0,0,255))
    #   #按照连通区域的索引来打上标签
    #   cv.putText(cv_imageBGR, str(i+1), (stat[0], stat[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
    # cv.imshow("Image window", cv_imageBGR)
    
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


def calculatepixelvalue(center_x, center_y, radius, img):
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


def drawcicle(img):
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

def savedata_csv(filename, header, data):
    with open(filename,"w") as datacsv:
      csvwriter = csv.writer(datacsv,delimiter=',')
      csvwriter.writerow(header)
      csvwriter.writerows(data)




def image_process():
    cv_image = cv.imread("image_raw_fibers.png")
        # test 
    # cv_image = drawcicle(cv_image)
    # gray = cv_image
    gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY ) 
    cv_imageBGR = cv_image # cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR ) 

    ##### max gradient threshold method
    h, w =gray.shape[:2]
    print(gray.shape)
    print("h=%d" %h)
    print("w=%d" %w)
    print("w*h=%d" %(w*h))
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
    circlenum = 24  #----------------------------defind circle number here
    circle_dected = empty([circlenum,3])
    for i, stat in enumerate(stats):
      if stat[4]>100 and stat[4]<1000:
        circle_dected[num,0] = int(stat[0] + stat[2]/2) # center_x
        circle_dected[num,1] = int(stat[1] + stat[3]/2) # center_y
        circle_dected[num,2] = int(stat[2]/2)+1 if stat[2]>stat[3] else int(stat[3]/2)+1 # radius
        num = num+1
    print("detected circle num = %d" %num)
    print(circle_dected)

    pixelvalue = []
    pixelnum = []
    circle_pixel = empty([circlenum,2])
    #### calculate total pixel value in each circle
    for i, stat in enumerate(circle_dected):
      pixelvalreturn, pixelnumreturn = calculatepixelvalue(stat[0], stat[1], stat[2], gray) 
      pixelvalue.append(pixelvalreturn)
      pixelnum.append(pixelnumreturn)
      circle_pixel[i,0] = pixelvalreturn
      circle_pixel[i,1] = pixelnumreturn
      print("%dth circle pixelvalue is: %d, pixelnum is %d" %(i+1,pixelvalreturn, pixelnumreturn))

    circle_stats = np.hstack((circle_dected,circle_pixel)) # center_x(column), center_y(row),radius,pixelvalue,pixelnum
    print(circle_stats)
    # np.savetxt('circle_stats.csv',circle_stats,delimiter=',')
    filename = 'cilcle_detected.csv'
    header = ["center_x(column)","center_y(row)","radius","pixelvalue","pixelnum"]
    savedata_csv(filename, header, circle_stats)

    cv.namedWindow("image_raw", cv.WINDOW_NORMAL)
    cv.imshow("image_raw", cv_image)  

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
    cv.imwrite("circled image.png", cv_imageBGR)



    # cv.waitKey()
    k = cv.waitKey(0)
    ## k = cv2.waitKey(0) & 0xFF  # 64位机器
    if k == 27:         # 按下esc时，退出
      cv.destroyAllWindows()

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
    # image_subscriber()
    image_process()



