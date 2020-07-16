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

Datamatrix = 0

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def gui():
    # define the form layout
    layout = [[sg.Text('Demo - Pressure Response',  justification='center', font='Helvetica 20')],
              [sg.Canvas(size=(1000, 800), key='-CANVAS-')],
              # [sg.Text('Force value and coordinates:',size=(40, 1),font='Helvetica 20'),
              [sg.Text('Force value and coordinates:',font='Helvetica 20'),
              sg.Text( 'loading:',font='Helvetica 20', key='-OUTPUT-')],
              # [[sg.Button('Connect',size=(10, 2), pad=((280, 20), 3), font='Helvetica 14')],
              [sg.Button('Connect',size=(10, 2), font='Helvetica 14'),
              sg.Button('Exit', size=(10, 2),  font='Helvetica 14')]]

    # create the form and show it without the plot
    window = sg.Window('Demo - Pressure Response', layout, finalize=True)

    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    # draw the intitial scatter plot
    fig = plt.figure(figsize=(10,8))

    ax = fig.add_subplot(111, projection='3d')
    # fig, ax = plt.subplots(111, projection='3d')
    ax.grid(True)
    fig_agg = draw_figure(canvas, fig)

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
        # for color in ['red', 'green', 'blue']:
        #     n = 750
        #     x, y = rand(2, n)
        #     scale = 200.0 * rand(n)
        #     ax.scatter(x, y, c=color, s=scale, label=color, alpha=0.3, edgecolors='none')
        R = np.sqrt(X ** 2 + Y ** 2)
        # Z = np.sin(R+Datamatrix)
        Z = np.sin(R+count)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
        ax.set_zlim3d(-1.01, 1.01)

    # ax.w_zaxis.set_major_locator(LinearLocator(10))
    # ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

        # fig.colorbar(surf, shrink=0.5, aspect=5) ## Show colorbar
        ax.legend()
        fig_agg.draw()
        count = count + 1
    window.close()



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
    a = threading.Thread(target = gui)
        # b = threading.Thread(target = printB)
    a.setDaemon(True)
    a.start()
    listener()