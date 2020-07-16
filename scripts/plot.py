#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from std_msgs.msg import Float64

import threading, time, signal
import sys

Datamatrix = 10.2

def draw():
    plt.ion()
    x = [0]
    y = [0] 
    while True:
        plt.clf()  
        x.append(Datamatrix)
        y.append(Datamatrix)
        plt.plot(x,y,'-r')
        # time.sleep(1) #time.sleep() does not work here, will show empty figure
        plt.pause(0.1)


####################### 3D plot with surface#######################
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# from matplotlib import cm
# # from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()

# ax = fig.add_subplot(1, 2, 1, projection='3d')
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
#                        linewidth=0, antialiased=False)
# ax.set_zlim3d(-1.01, 1.01)

# # ax.w_zaxis.set_major_locator(LinearLocator(10))
# # ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)
###################################################################

# def printB():
#     while True:
#         print 'b'
#         time.sleep(1)

def quit(signum, frame):
    print('You choose to stop me.')
    sys.exit()



def callback(data):
    global Datamatrix
    Datamatrix = data.data
    rospy.loginfo("I heard %0.6f", data.data)
    
    

    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    print("listener starts")
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("matrix", Float64, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    # listener()
    try:
        signal.signal(signal.SIGINT, quit)
        signal.signal(signal.SIGTERM, quit)
        a = threading.Thread(target = draw)
        # b = threading.Thread(target = printB)
        a.setDaemon(True)
        a.start()
        listener()
        # b.setDaemon(True)
        # b.start()

        while True:
            pass
    except(Exception,exc):
        print(exc)


