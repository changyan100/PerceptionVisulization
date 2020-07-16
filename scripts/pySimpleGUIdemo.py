#!/usr/bin/env python
# -*- coding: utf-8 -*-
import PySimpleGUI as sg
from random import randint
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.figure import Figure

# Yet another usage of MatPlotLib with animations.

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def main():

    NUM_DATAPOINTS = 10000
    # define the form layout
    layout = [[sg.Text('Animated Matplotlib', size=(40, 1),
                justification='center', font='Helvetica 20')],
              [sg.Canvas(size=(800, 600), key='-CANVAS-')],
              [sg.Text('Progress through the data')],
              [sg.Slider(range=(0, NUM_DATAPOINTS), size=(60, 10),
                orientation='h', key='-SLIDER-')],
              [sg.Text('Number of data points to display on screen')],
               [sg.Slider(range=(10, 500), default_value=40, size=(40, 10),
                    orientation='h', key='-SLIDER-DATAPOINTS-')],
              [sg.Button('Exit', size=(10, 1), pad=((280, 0), 3), font='Helvetica 14')]]

    # create the form and show it without the plot
    window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI',
                layout, finalize=True)

    canvas_elem = window['-CANVAS-']
    slider_elem = window['-SLIDER-']
    canvas = canvas_elem.TKCanvas

    # draw the initial plot in the window
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.grid()
    fig_agg = draw_figure(canvas, fig)
    # make a bunch of random data points
    dpts = [randint(0, 10) for x in range(NUM_DATAPOINTS)]

    for i in range(len(dpts)):

        event, values = window.read(timeout=10)
        if event in ('Exit', None):
            exit(69)
        slider_elem.update(i)       # slider shows "progress" through the data points
        ax.cla()                    # clear the subplot
        ax.grid()                   # draw the grid
        data_points = int(values['-SLIDER-DATAPOINTS-']) # draw this many data points (on next line)
        ax.plot(range(data_points), dpts[i:i+data_points],  color='purple')
        fig_agg.draw()

    window.close()

if __name__ == '__main__':
    main()



###############################################3


# sg.popup('Hello From PySimpleGUI!', 'This is the shortest GUI program ever!')

###############################################################################
# sg.theme('Dark Blue 1')  # please make your creations colorful

# layout = [  [sg.Text('Filename')],
#             [sg.Input(), sg.FileBrowse()], 
#             [sg.OK(), sg.Cancel()],
#             [sg.Input()] ] 

# window = sg.Window('Get filename example', layout)

# event, values = window.read()

# window.close()

########################################################################

# event, values = sg.Window('Get filename example', 
# 	[[sg.Text('Filename')], [sg.Input(), sg.FileBrowse()], 
# 	[sg.OK(), sg.OK(), sg.Cancel()] ]).read(close=True)
 
# # Create some widgets
# text = sg.Text("What's your name?")
# text_entry = sg.InputText()
# ok_btn = sg.Button('OK')
# cancel_btn = sg.Button('Cancel')
# layout = [[text, text_entry],
#           [ok_btn, cancel_btn]]
 
# # Create the Window
# window = sg.Window('Hello PySimpleGUI', layout)
 
# # Create the event loop
# while True:
#     event, values = window.read()
#     if event in (None, 'Cancel'):
#         # User closed the Window or hit the Cancel button
#         break
#     print(f'Event: {event}')
#     print(str(values))
 
# window.close()

##############################################################

# sg.theme('DarkAmber')   # Add a touch of color
# # All the stuff inside your window.
# layout = [  [sg.Text('Some text on Row 1')],
#             [sg.Text('Enter something on Row 2'), sg.InputText()],
#             [sg.Button('Ok'), sg.Button('Cancel')] ]

# # Create the Window
# window = sg.Window('Window Title', layout)
# # Event Loop to process "events" and get the "values" of the inputs
# while True:
#     event, values = window.read()
#     if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
#         break
#     print('You entered ', values[0])

# window.close()
###################################################################