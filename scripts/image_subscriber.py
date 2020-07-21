#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from std_msgs.msg import Float64

import threading, time, signal
import sys
from PIL import Image
import PIL
from sensor_msgs.msg import Image as image_msg

import cv2
from cv_bridge import CvBridge, CvBridgeError

# def draw():
#     plt.ion()
#     x = [0]
#     y = [0] 
#     while True:
#         plt.clf()  
#         x.append(Datamatrix)
#         y.append(Datamatrix)
#         plt.plot(x,y,'-r')
#         # time.sleep(1) #time.sleep() does not work here, will show empty figure
#         plt.pause(0.1)


# def quit(signum, frame):
#     print('You choose to stop me.')
#     sys.exit()



def callback(data):
    # global image_raw
    # image_raw = data.data
    # rospy.loginfo("I heard %0.6f", data.data)
    rospy.loginfo("reading image...")
    print(data.data)


    # image_raw = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    # im = Image.fromarray(image_raw)
    # im.show()
    
    

    
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


