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


from std_msgs.msg import String
from sensor_msgs.msg import Image as image_msg
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
# from PIL import Image

from visulization.msg import IntList
import math

import pylab
from mpl_toolkits.mplot3d import proj3d


# datareceived = []
# cv_image = []
exitflag = 1
countgloabal = 1
tolorance = 6000

# use matplotlib plot wireframe in callback, to synchrinize data

def callback(data):

  # fiberfor_x = args[0]
  # fiberfor_y = args[1]

  # print("fiber index for x axis are:", fiberfor_x)
  # print("fiber index for y axis are:", fiberfor_y)

  global tolorance
  # global cv_image
  # # Datamatrix = data.data
  # bridge = CvBridge()
  # # rospy.loginfo("I heard %0.6f", data.data)
  # try:
  #   # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
  #   cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
  # except CvBridgeError as e:
  #   print(e)
  # cv.namedWindow("Image window", cv.WINDOW_NORMAL)
  # cv.imshow("Image window", cv_image)
  # cv.waitKey(3)

  datareceived = data.data
  #datareceived = [fibernum, index1, index2, changed value1, changed value2]
  print("I received:", datareceived)

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


  if datareceived:

    
    # axisrange = datareceived[0]+1

    # X = np.arange(1, axisrange, griddense)
    # Y = np.arange(1, axisrange, griddense)

    xx = np.arange(0, datareceived[-2]+1+griddense, griddense)  #say datareceived[-2] = 2, then xx = [0,1,2,3]
    yy = np.arange(0, datareceived[-1]+1+griddense, griddense)

    X, Y = np.meshgrid(xx, yy)  #meshgrid make X and Y have same dimension: row = len(yy), colounm = len(xx)
    # datalen = int((axisrange-1)/griddense)
    # Z = np.zeros([datalen, datalen])
    Z = np.zeros([yy.shape[0],xx.shape[0]])  
    
    if abs(datareceived[3])>tolorance:

      # index_x = int((datareceived[1])/griddense)
      # index_y = int((datareceived[2])/griddense)

      # if datareceived[1] in fiberfor_x:
      #   index_x = fiberfor_x.index(datareceived[1])+1
        
      # elif datareceived[1] in fiberfor_y:
      #   index_y = fiberfor_y.index(datareceived[1])+1
        
      # else:
      #   print("\033[0;37;40m\tdetected fiber index doesnot match any axis\033[0m")

      # if datareceived[2] in fiberfor_x:
      #   index_x = fiberfor_x.index(datareceived[2])+1
        
      # elif datareceived[2] in fiberfor_y:
      #   index_y = fiberfor_y.index(datareceived[2])+1
        
      # else:
      #   print("\033[0;37;40m\tdetected fiber index doesnot match any axis\033[0m")
      
      # if 'index_x' in locals():
      #   print("index_x before nomalization: ", index_x)
      #   index_x = int(index_x/griddense)
      #   print("index_x after nomalization: ", index_x)
      #   print("index_x-cut_len/2:", index_x-cut_len/2)

      # if 'index_y' in locals():
      #   print("index_y before nomalization: ", index_y)
      #   index_y = int(index_y/griddense)
      #   print("index_y after nomalization: ", index_y)
      #   print("index_y-cut_len/2:", index_y-cut_len/2)

      # Z[index_x, index_y] = datareceived[3]/100
      # datareceived[1] = datareceived[1]+1
      # datareceived[2] = datareceived[2]+1

      # Z_cut_current = np.zeros([cut_len, cut_len])
      detI = datareceived[3]/100

      detI = 10
      Z_cut_current = detI*Z_cut  



      # if 'index_x' in locals() and 'index_y' in locals():
      #   Z[int(index_x-cut_len/2):int(index_x+cut_len/2),int(index_y-cut_len/2):int(index_y+cut_len/2)] = Z_cut_current
      index_x = datareceived[1]
      index_y = datareceived[2]
      index_x = int(index_x/griddense)
      index_y = int(index_y/griddense)
      Z[int(index_x-int(cut_len/2)):int(index_x+int(cut_len/2))+1,int(index_y-int(cut_len/2)):int(index_y+int(cut_len/2))+1] = Z_cut_current

  
  

  else: #no data received from iamge process
    X = np.arange(1, 5, 0.1)
    Y = np.arange(1, 5, 0.1)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
      # Z = np.sin(R+Datamatrix)
    Z = np.sin(R)

  wf = plt.axes(projection = '3d')
  wf.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

  if 'index_x' in locals() and 'index_y' in locals():
    x2, y2, _ = proj3d.proj_transform(yy[-index_y],xx[index_x],detI-1, wf.get_proj())
    # x2, y2, _ = proj3d.proj_transform(index_x,index_y,detI*100, wf.get_proj())
    display = 'coordinates = '+str((datareceived[1], datareceived[2]))
    label = pylab.annotate(
        display, 
        xy = (x2, y2), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

  plt.gca().invert_xaxis()
  # wf.zlim(0, 15) # 或 xlim((left, right))
  wf.set_zlim3d(0,11)

  plt.draw()
  plt.pause(0.0000001)





def listener():

  # In ROS, nodes are uniquely named. If two nodes with the same
  # name are launched, the previous one is kicked off. The
  # anonymous=True flag means that rospy will choose a unique
  # name for our 'listener' node so that multiple listeners can
  # run simultaneously.
  print("Program starts")
  # fiberfor_x = []
  # fiberfor_y = []
  # while True:
  #   char = input("Please input fiber index for X axis or [E/e] to continue: ")
  #   if char == 'e' or char == 'E':
  #     break
  #   else:
  #     fiberfor_x.append(int(char))

  # print("fiber index for x axis are:", fiberfor_x)

  # while True:
  #   char = input("Please input fiber index for Y axis or [E/e] to continue: ")
  #   if char == 'e' or char == 'E':
  #     break
  #   else:
  #     fiberfor_y.append(int(char))
  

  
  rospy.init_node('GUI_listener', anonymous=True)
  rospy.Subscriber("fiber_index", IntList, callback)
  # spin() simply keeps python from exiting until this node is stopped
  plt.ion()
  plt.show()
  rospy.spin()



if __name__ == '__main__':
    listener()



'''
# write gui plot into callback, doesnot work, error: main thread in not in main loop

def draw_figure(canvas_plot, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas_plot)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def callback(data,args):
    # global cv_image
    # # Datamatrix = data.data
    # bridge = CvBridge()
    # # rospy.loginfo("I heard %0.6f", data.data)
    # try:
    #   # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    #   cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
    # except CvBridgeError as e:
    #   print(e)
    # cv.namedWindow("Image window", cv.WINDOW_NORMAL)
    # cv.imshow("Image window", cv_image)
    # cv.waitKey(3)

    # global exitflag
    # global datareceived
    # global countgloabal

    datareceived = data.data
    print("I received:", datareceived)

    ax = args[0]
    fig_agg = args[1]
    window = args[2]

    #############################################update wireframe range here
    # global datareceived

    # global exitflag

    count = 0

    griddense = 0.1
    # define a small range to show spike signal nicely with cos function
    R = 0.5
    x_cut = np.arange(-R,R,griddense)
    y_cut = np.arange(-R,R,griddense)
    X_cut, Y_cut = np.meshgrid(x_cut,y_cut)
    r = np.sqrt(X_cut**2 + Y_cut**2)
    cut_len = int(2*R/griddense)
    Z_cut = np.zeros([cut_len, cut_len])
    for i in range(Z_cut.shape[0]):
      for j in range(Z_cut.shape[1]):
        Z_cut[i,j] = np.cos(math.pi*r[i,j]) if r[i,j]<=R else 0


    # print("I am in while loop")
    if datareceived:

      axisrange = datareceived[0]+1

      X = np.arange(1, axisrange, griddense)
      Y = np.arange(1, axisrange, griddense)
      X, Y = np.meshgrid(X, Y)
      datalen = int((axisrange-1)/griddense)
      Z = np.zeros([datalen, datalen])

      index_x = int((datareceived[1])/griddense)
      index_y = int((datareceived[2])/griddense)
      # Z[index_x, index_y] = datareceived[3]/100
      # datareceived[1] = datareceived[1]+1
      # datareceived[2] = datareceived[2]+1
      Z_cut_current = np.zeros([cut_len, cut_len])
      detI = datareceived[3]/100
      Z_cut_current = detI*Z_cut  #+detI
      # Z_cut_current = Z_cut+detI
      # print("index_y:", index_y)
      # print("index_y-cut_len/2:", index_y-cut_len/2)

      Z[int(index_x-cut_len/2):int(index_x+cut_len/2),int(index_y-cut_len/2):int(index_y+cut_len/2)] = Z_cut_current

      # window['-OUTPUT-'].update(str([datareceived[1]+1,datareceived[2]+1,datareceived[3],datareceived[4]])) ####################update displayed data here


    else:
      X = np.arange(1, 5, 0.1)
      Y = np.arange(1, 5, 0.1)
      X, Y = np.meshgrid(X, Y)
      R = np.sqrt(X ** 2 + Y ** 2)
        # Z = np.sin(R+Datamatrix)
      Z = np.sin(R+count)

      window['-OUTPUT-'].update('Loanding') ####################update displayed data here


    # event, values = window.read(timeout=10)
    # if event in ('Exit', None):
    #     # break
    #   exitflag = 0
    #   exit(69)
       
    # if event in ('Connect', None):
    #   sg.popup_ok('Successful Connection')  # Shows OK button

      
    ax.cla()
    ax.grid(True)
    # R = np.sqrt(X ** 2 + Y ** 2)
    # # Z = np.sin(R+Datamatrix)
    # Z = np.sin(R+count)
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
    #                    linewidth=0, antialiased=False)
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    # ax.set_zlim3d(-1.01, 1.01)

# ax.w_zaxis.set_major_locator(LinearLocator(10))
# ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5) ## Show colorbar
    ax.legend()
    fig_agg.draw()
    count = count + 1



def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    sg.theme('LightGreen')
    figure_w = 650
    figure_h = 650
    # define the form layout
    layout = [[sg.Text('Demo - Pressure Response',  justification='center', font='Helvetica 20')],
              [sg.Canvas(size=(figure_w, figure_h), key='-CANVAS_plot-'), 
              # sg.Canvas(size=(figure_w, figure_h), key='-CANVAS_image-') ,
              ],
              # [sg.Text('Force value and coordinates:',size=(40, 1),font='Helvetica 20'),
              [sg.Text('Force value and coordinates:',font='Helvetica 20'),
              sg.Text( 'loading:',font='Helvetica 20',size=(40, 1), key='-OUTPUT-')],
              # [[sg.Button('Connect',size=(10, 2), pad=((280, 20), 3), font='Helvetica 14')],
              [sg.Button('Connect',size=(10, 2), font='Helvetica 14'),
              sg.Button('Exit', size=(10, 2),  font='Helvetica 14')]]

    # create the form and show it without the plot
    window = sg.Window('Demo - Pressure Response', layout, resizable=True, finalize=True)

    # image_elem = window['-CANVAS_image-']

    canvas_plot_elem = window['-CANVAS_plot-']
    canvas_plot = canvas_plot_elem.TKCanvas
    # draw the intitial scatter plot
    fig = plt.figure(figsize=(8,6))

    ax = fig.add_subplot(111, projection='3d')
    # fig, ax = plt.subplots(111, projection='3d')
    # ax.grid(True)
    fig_agg = draw_figure(canvas_plot, fig)
    


    rospy.init_node('GUI_listener', anonymous=True)
    # rospy.Subscriber("image_raw", image_msg, callback)
    rospy.Subscriber("fiber_index", IntList, callback, (ax,fig_agg, window))
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    window.close()


if __name__ == '__main__':
    listener()
'''



'''
# make a thread for GUI and keep listerner as it is

datareceived = []
cv_image = []
exitflag = 1
countgloabal = 1

def draw_figure(canvas_plot, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas_plot)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def gui():
    global countgloabal
    sg.theme('LightGreen')
    figure_w = 650
    figure_h = 650
    # define the form layout
    layout = [[sg.Text('Demo - Pressure Response',  justification='center', font='Helvetica 20')],
              [sg.Canvas(size=(figure_w, figure_h), key='-CANVAS_plot-'), 
              # sg.Canvas(size=(figure_w, figure_h), key='-CANVAS_image-') ,
              ],
              # [sg.Text('Force value and coordinates:',size=(40, 1),font='Helvetica 20'),
              [sg.Text('Force value and coordinates:',font='Helvetica 20'),
              sg.Text( 'loading:',font='Helvetica 20',size=(40, 1), key='-OUTPUT-')],
              # [[sg.Button('Connect',size=(10, 2), pad=((280, 20), 3), font='Helvetica 14')],
              [sg.Button('Connect',size=(10, 2), font='Helvetica 14'),
              sg.Button('Exit', size=(10, 2),  font='Helvetica 14')]]

    # create the form and show it without the plot
    window = sg.Window('Demo - Pressure Response', layout, resizable=True, finalize=True)

    # image_elem = window['-CANVAS_image-']

    canvas_plot_elem = window['-CANVAS_plot-']
    canvas_plot = canvas_plot_elem.TKCanvas
    # draw the intitial scatter plot
    fig = plt.figure(figsize=(8,6))

    ax = fig.add_subplot(111, projection='3d')
    # fig, ax = plt.subplots(111, projection='3d')
    # ax.grid(True)

    fig_agg = draw_figure(canvas_plot, fig)
    #############################################update wireframe range here
    global datareceived

    global exitflag

    count = 0

    griddense = 0.1
    # define a small range to show spike signal nicely with cos function
    R = 0.5
    x_cut = np.arange(-R,R,griddense)
    y_cut = np.arange(-R,R,griddense)
    X_cut, Y_cut = np.meshgrid(x_cut,y_cut)
    r = np.sqrt(X_cut**2 + Y_cut**2)
    cut_len = int(2*R/griddense)
    Z_cut = np.zeros([cut_len, cut_len])
    for i in range(Z_cut.shape[0]):
      for j in range(Z_cut.shape[1]):
        Z_cut[i,j] = np.cos(math.pi*r[i,j]) if r[i,j]<=R else 0


    while True:
        # print("I am in while loop")
        if datareceived:

          axisrange = datareceived[0]+1

          X = np.arange(1, axisrange, griddense)
          Y = np.arange(1, axisrange, griddense)
          X, Y = np.meshgrid(X, Y)
          datalen = int((axisrange-1)/griddense)
          Z = np.zeros([datalen, datalen])

          index_x = int((datareceived[1])/griddense)
          index_y = int((datareceived[2])/griddense)
          # Z[index_x, index_y] = datareceived[3]/100
          # datareceived[1] = datareceived[1]+1
          # datareceived[2] = datareceived[2]+1
          Z_cut_current = np.zeros([cut_len, cut_len])
          detI = datareceived[3]/100
          Z_cut_current = detI*Z_cut  #+detI
          # Z_cut_current = Z_cut+detI
          # print("index_y:", index_y)
          # print("index_y-cut_len/2:", index_y-cut_len/2)

          Z[int(index_x-cut_len/2):int(index_x+cut_len/2),int(index_y-cut_len/2):int(index_y+cut_len/2)] = Z_cut_current

        else:
          X = np.arange(1, 5, 0.1)
          Y = np.arange(1, 5, 0.1)
          X, Y = np.meshgrid(X, Y)
          R = np.sqrt(X ** 2 + Y ** 2)
            # Z = np.sin(R+Datamatrix)
          Z = np.sin(R+count)


        event, values = window.read(timeout=10)
        if event in ('Exit', None):
            # break
          exitflag = 0
          exit(69)
           
        if event in ('Connect', None):
          sg.popup_ok('Successful Connection')  # Shows OK button
        
        if datareceived:
          window['-OUTPUT-'].update(str([datareceived[1]+1,datareceived[2]+1,datareceived[3],datareceived[4]])) ####################update displayed data here
        else:
          window['-OUTPUT-'].update('Loanding') ####################update displayed data here



        ax.cla()
        ax.grid(True)
        # R = np.sqrt(X ** 2 + Y ** 2)
        # # Z = np.sin(R+Datamatrix)
        # Z = np.sin(R+count)
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        #                    linewidth=0, antialiased=False)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        # ax.set_zlim3d(-1.01, 1.01)

    # ax.w_zaxis.set_major_locator(LinearLocator(10))
    # ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

        # fig.colorbar(surf, shrink=0.5, aspect=5) ## Show colorbar
        ax.legend()
        fig_agg.draw()
        count = count + 1

        print("count in GUI is:", countgloabal)

    # show camera image
        # imgbytes = cv.imencode('.png', cv_image)[1].tobytes()  # ditto

        # img = Image.fromarray(cv_image)    # create PIL image from frame
        # bio = io.BytesIO()              # a binary memory resident stream
        # img.save(bio, format= 'PNG')    # save image as png to it
        # imgbytes = bio.getvalue()       # this can be used by OpenCV hopefully


        # image_elem.update(data=imgbytes)
    window.close()



def callback(data):
    # global cv_image
    # # Datamatrix = data.data
    # bridge = CvBridge()
    # # rospy.loginfo("I heard %0.6f", data.data)
    # try:
    #   # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    #   cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
    # except CvBridgeError as e:
    #   print(e)
    # cv.namedWindow("Image window", cv.WINDOW_NORMAL)
    # cv.imshow("Image window", cv_image)
    # cv.waitKey(3)
    global exitflag
    global datareceived
    global countgloabal

    datareceived = data.data
    # print("I received:", datareceived)
    print("count = ", countgloabal)
    countgloabal = countgloabal+1

    if exitflag==0:
      rospy.signal_shutdown("Program aborted!")

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # np.set_printoptions(suppress=True)
    # print("listener starts")
    # filename = '2020-07-29-15_13_05 detected circle stats.csv'
    # data_init = np.loadtxt(filename,delimiter=",", skiprows=1)
    # print("file name is ", filename)
    # print("data read from the initialization file is:")
    # print(data_init)

    rospy.init_node('GUI_listener', anonymous=True)
    # rospy.Subscriber("image_raw", image_msg, callback)
    rospy.Subscriber("fiber_index", IntList, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    
    # listener()
    a = threading.Thread(target = gui)
        # b = threading.Thread(target = printB)
    # a.setDaemon(True)
    
    # b = threading.Thread(target = listener)
        # b = threading.Thread(target = printB)
    a.setDaemon(True)
    a.start()
    listener()

'''


'''
class visulizationGUI:
    def __init__(self):

        

        # sg.theme('LightGreen')
        # figure_w = 650
        # figure_h = 650
        # # define the form layout
        # layout = [[sg.Text('Demo - Pressure Response',  justification='center', font='Helvetica 20')],
        #           [sg.Canvas(size=(figure_w, figure_h), key='-CANVAS_plot-'), sg.Canvas(size=(figure_w, figure_h), key='-CANVAS_image-') ,],
        #           # [sg.Text('Force value and coordinates:',size=(40, 1),font='Helvetica 20'),
        #           [sg.Text('Force value and coordinates:',font='Helvetica 20'),
        #           sg.Text( 'loading:',font='Helvetica 20', key='-OUTPUT-')],
        #           # [[sg.Button('Connect',size=(10, 2), pad=((280, 20), 3), font='Helvetica 14')],
        #           [sg.Button('Connect',size=(10, 2), font='Helvetica 14'),
        #           sg.Button('Exit', size=(10, 2),  font='Helvetica 14')]]

        # # create the form and show it without the plot
        # self.window = sg.Window('Demo - Pressure Response', layout, resizable=True, finalize=True)

        # image_elem = self.window['-CANVAS_image-']

        # canvas_plot_elem = self.window['-CANVAS_plot-']
        # canvas_plot = canvas_plot_elem.TKCanvas
        # # draw the intitial scatter plot
        # fig = plt.figure(figsize=(8,6))

        # self.ax = fig.add_subplot(111, projection='3d')
        # # fig, ax = plt.subplots(111, projection='3d')
        # self.ax.grid(True)
        # self.fig_agg = self.draw_figure(canvas_plot, fig)

        # X = np.arange(-5, 5, 0.25)
        # Y = np.arange(-5, 5, 0.25)
        # X, Y = np.meshgrid(X, Y)

        print("I am in Init")

        # self.sub = rospy.Subscriber("image_raw", image_msg, callback)
        self.sub = rospy.Subscriber("fiber_index", IntList, self.callback)

        print("I am in Init end")

    def draw_figure(self, canvas_plot, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas_plot)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg


    def callback(self, data):
        # global cv_image
        # # Datamatrix = data.data
        # bridge = CvBridge()
        # # rospy.loginfo("I heard %0.6f", data.data)
        # try:
        #   # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        #   cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
        # except CvBridgeError as e:
        #   print(e)
        # cv.namedWindow("Image window", cv.WINDOW_NORMAL)
        # cv.imshow("Image window", cv_image)
        # cv.waitKey(3)
        print("I am in callback")
        print("I received:", data)

    #     event, values = self.window.read(timeout=10)
    #     if event in ('Exit', None):
    #         exit(69)
    #     if event in ('Connect', None):
    #         sg.popup_ok('Successful Connection')  # Shows OK button
    #     self.window['-OUTPUT-'].update('null')

    #     self.ax.cla()
    #     self.ax.grid(True)
    #     R = np.sqrt(X ** 2 + Y ** 2)
    #     # Z = np.sin(R+Datamatrix)
    #     Z = np.sin(R+count)
    #     # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
    #     #                    linewidth=0, antialiased=False)
    #     self.ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    #     self.ax.set_zlim3d(-1.01, 1.01)

    # # ax.w_zaxis.set_major_locator(LinearLocator(10))
    # # ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

    #     # fig.colorbar(surf, shrink=0.5, aspect=5) ## Show colorbar
    #     self.ax.legend()
    #     self.fig_agg.draw()
    #     count = count + 1
        self.gui(data.data)





    def gui(self,data):
        sg.theme('LightGreen')
        figure_w = 650
        figure_h = 650
        # define the form layout
        layout = [[sg.Text('Demo - Pressure Response',  justification='center', font='Helvetica 20')],
                  [sg.Canvas(size=(figure_w, figure_h), key='-CANVAS_plot-'), sg.Canvas(size=(figure_w, figure_h), key='-CANVAS_image-') ,],
                  # [sg.Text('Force value and coordinates:',size=(40, 1),font='Helvetica 20'),
                  [sg.Text('Force value and coordinates:',font='Helvetica 20'),
                  sg.Text( 'loading:',font='Helvetica 20', key='-OUTPUT-')],
                  # [[sg.Button('Connect',size=(10, 2), pad=((280, 20), 3), font='Helvetica 14')],
                  [sg.Button('Connect',size=(10, 2), font='Helvetica 14'),
                  sg.Button('Exit', size=(10, 2),  font='Helvetica 14')]]

        # create the form and show it without the plot
        window = sg.Window('Demo - Pressure Response', layout, resizable=True, finalize=True)

        image_elem = window['-CANVAS_image-']

        canvas_plot_elem = window['-CANVAS_plot-']
        canvas_plot = canvas_plot_elem.TKCanvas
        # draw the intitial scatter plot
        fig = plt.figure(figsize=(8,6))

        ax = fig.add_subplot(111, projection='3d')
        # fig, ax = plt.subplots(111, projection='3d')
        ax.grid(True)
        fig_agg = self.draw_figure(canvas_plot, fig)

        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)

        count = 0
        # while True:
        event, values = window.read(timeout=10)
        if event in ('Exit', None):
            exit(69)
        if event in ('Connect', None):
            sg.popup_ok('Successful Connection')  # Shows OK button
        window['-OUTPUT-'].update(str(data[0]))

        ax.cla()
        ax.grid(True)
        R = np.sqrt(X ** 2 + Y ** 2)
        # Z = np.sin(R+Datamatrix)
        Z = np.sin(R+count)
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        #                    linewidth=0, antialiased=False)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        ax.set_zlim3d(-1.01, 1.01)

    # ax.w_zaxis.set_major_locator(LinearLocator(10))
    # ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

        # fig.colorbar(surf, shrink=0.5, aspect=5) ## Show colorbar
        ax.legend()
        fig_agg.draw()
        # count = count + 1

        # show camera image
            # imgbytes = cv.imencode('.png', cv_image)[1].tobytes()  # ditto

            # img = Image.fromarray(cv_image)    # create PIL image from frame
            # bio = io.BytesIO()              # a binary memory resident stream
            # img.save(bio, format= 'PNG')    # save image as png to it
            # imgbytes = bio.getvalue()       # this can be used by OpenCV hopefully


            # image_elem.update(data=imgbytes)
        # window.close()



# def callback(data):
#     global cv_image
#     # Datamatrix = data.data
#     bridge = CvBridge()
#     # rospy.loginfo("I heard %0.6f", data.data)
#     try:
#       # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
#       cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
#     except CvBridgeError as e:
#       print(e)
#     cv.namedWindow("Image window", cv.WINDOW_NORMAL)
#     cv.imshow("Image window", cv_image)
#     cv.waitKey(3)


# def listener():

#     # In ROS, nodes are uniquely named. If two nodes with the same
#     # name are launched, the previous one is kicked off. The
#     # anonymous=True flag means that rospy will choose a unique
#     # name for our 'listener' node so that multiple listeners can
#     # run simultaneously.
#     # np.set_printoptions(suppress=True)
#     # print("listener starts")
#     # filename = '2020-07-29-15_13_05 detected circle stats.csv'
#     # data_init = np.loadtxt(filename,delimiter=",", skiprows=1)
#     # print("file name is ", filename)
#     # print("data read from the initialization file is:")
#     # print(data_init)

#     rospy.init_node('GUI_image_listener', anonymous=True)
#     rospy.Subscriber("image_raw", image_msg, callback)
#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()


def main(args):
  
  GUI = visulizationGUI()
  rospy.init_node('GUI_subscriber', anonymous=True)
  # rospy.init_node('image_converter', anonymous=True)
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


# if __name__ == '__main__':
    
#     # # listener()
#     # a = threading.Thread(target = gui)
#     #     # b = threading.Thread(target = printB)
#     # a.setDaemon(True)
#     # a.start()
#     # listener()

'''