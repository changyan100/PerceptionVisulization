#!/usr/bin/env python

'''
Visulization GUI, plot wireframe with realtime fiber index, which is subscribed in topic "fiber_index"
'''

import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from numpy.random import rand

import rospy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np
from std_msgs.msg import Float64

import threading, time, signal
import sys


from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d


from std_msgs.msg import String
from sensor_msgs.msg import Image as image_msg
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
# from PIL import Image

from visulization.msg import IntList
import math

import pylab
from mpl_toolkits.mplot3d import proj3d

def test():
	
  griddense = 0.05
  # define a small range to show spike signal nicely with cos function
  R = 0.5
  x_cut = np.arange(-R,R+griddense,griddense)
  y_cut = np.arange(-R,R+griddense,griddense)
  X_cut, Y_cut = np.meshgrid(x_cut,y_cut)
  r = np.sqrt(X_cut**2 + Y_cut**2)
  cut_len = x_cut.shape[0]
  Z_cut = np.zeros([cut_len, cut_len])
  for i in range(Z_cut.shape[0]):
    for j in range(Z_cut.shape[1]):
      Z_cut[i,j] = 0.5*(np.cos(math.pi/R*r[i,j])+1) if r[i,j]<=R else 0
  Z_cut = 100*Z_cut

  # wf = plt.axes(projection = '3d')
  # wf.plot_wireframe(X_cut, Y_cut, Z_cut, rstride=1, cstride=1,color='green')


  # # x2, y2, _ = proj3d.proj_transform(index_x,index_y,10, wf.get_proj())
  # # x2, y2, _ = proj3d.proj_transform(index_x,index_y,detI*100, wf.get_proj())

  # # label = pylab.annotate(
  # #     "this", 
  # #     xy = (x2, y2), xytext = (-20, 20),
  # #     textcoords = 'offset points', ha = 'right', va = 'bottom',
  # #     bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
  # #     arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


  # plt.draw()
  # plt.show()
  print(X_cut)

  print(Y_cut)

  print(Z_cut)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  # plt.tick_params(axis='both', which='major', direction='out')

  # ax.xaxis.set_ticks_position('top') #将x轴的位置设置在顶部
  
  
  # Plot a basic wireframe.
  ax.plot_wireframe(X_cut, Y_cut, Z_cut, rstride=1, cstride=1)

  plt.xlim(-1, 1) # 或 xlim((left, right))

  ax = plt.gca()
  ax.invert_xaxis()
  plt.show()

if __name__ == '__main__':
    test()
