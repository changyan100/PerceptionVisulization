#!/usr/bin/env python
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from numpy.random import rand

import rospy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
from PIL import Image

Datamatrix = 0
cv_image = []

def draw_figure(canvas_plot, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas_plot)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def gui():
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
    fig_agg = draw_figure(canvas_plot, fig)

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)

    count = 0
    while True:
        event, values = window.read(timeout=10)
        if event in ('Exit', None):
            exit(69)
        if event in ('Connect', None):
            sg.popup_ok('Successful Connection')  # Shows OK button
        window['-OUTPUT-'].update('null')

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
        count = count + 1

    # show camera image
        # imgbytes = cv.imencode('.png', cv_image)[1].tobytes()  # ditto

        # img = Image.fromarray(cv_image)    # create PIL image from frame
        # bio = io.BytesIO()              # a binary memory resident stream
        # img.save(bio, format= 'PNG')    # save image as png to it
        # imgbytes = bio.getvalue()       # this can be used by OpenCV hopefully


        # image_elem.update(data=imgbytes)
    window.close()



def callback(data):
    global cv_image
    # Datamatrix = data.data
    bridge = CvBridge()
    # rospy.loginfo("I heard %0.6f", data.data)
    try:
      # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # output cv:mat
    except CvBridgeError as e:
      print(e)
    cv.imshow("Image window", cv_image)
    cv.waitKey(3)


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    print("listener starts")
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("image_raw", image_msg, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    
    # listener()
    a = threading.Thread(target = gui)
        # b = threading.Thread(target = printB)
    a.setDaemon(True)
    a.start()
    listener()