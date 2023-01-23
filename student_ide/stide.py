#!/usr/bin/python
import PySimpleGUI as sg
import re
import sys
import os
import threading
import requests



def clear():
    window['-TEXT-'].update("")


def post_data():

    url = "http://127.0.0.1:5001/student"
    headers = {'content-type': "multipart/form-data"}
    directory = "./dataset"
    if len(os.listdir(directory))==0:
        return
    file_dict = {}
    for filename in os.listdir(directory):
        with open(directory+"/"+filename, 'rb') as f:
            file_dict[filename] = f.read()
    r = requests.post(url, files=file_dict)
    if r.status_code != 200:
        print("Failed to send your code")
    return

if not os.path.exists("dataset"): os.mkdir("dataset")
for f in os.listdir("dataset"):
    os.remove(os.path.join("dataset", f))
roll_no = ""
counter = 0
prog = 1
    
var = sys.stdout
w_wid = 50
w_hei = 50

f_run = "Run .. (Ctrl+R)"
f_prog = "New Program ... (Ctrl+N)"
menu_layout = [['Run', [f_run]],['New', [f_prog]]]


layout = [[sg.Menu(menu_layout)],[sg.Text(size=(24,3), key='-TEXT-')],[sg.Text(size=(24,10), key='-TEXT-')], [sg.Multiline(font=('Arial',15), size=(w_wid,w_hei), key='_BODY_')]]

layout_inp = [
    [sg.Text('Please enter your Roll No')],
    [sg.Text('Roll Number', size =(15, 1)), sg.InputText()],
    [sg.Submit(bind_return_key = False,)]
]

window = sg.Window('PY IDE', layout=layout, margins=(0, 0), resizable=True, return_keyboard_events=True)#,web_port=2222)


window_inp = sg.Window('PY INP', layout = layout_inp, resizable=True, return_keyboard_events=True).Finalize()
window_inp['Submit'].update(disabled=True)

while(True):
    event,values = window_inp.read()
    if values[0] != '' and re.match('[0-9]{12}$', values[0]):
        window_inp['Submit'].update(disabled=False)
    else:
        window_inp['Submit'].update(disabled=True)
    
    if(event in (None,'Exit')):
        exit(0)
    if event == 'Submit':
        roll_no = values[0]
        window_inp.close()
        break


window.Finalize()
window.Maximize()
w_win,h_win = window.size
_,h_body = window['_BODY_'].Size
layout = [[sg.Menu(menu_layout)],[sg.Text(size=(24,8), key='-TEXT-')], [sg.Text('Enter comma seperated inputs',size=(24,1))], [sg.Multiline(font=('Arial',15), size=(w_win,2), key='_INP_')],[sg.Multiline(font=('Arial',15), size=(w_win,h_body), key='_BODY_')]]
window.close()
window = sg.Window('PY IDE', layout=layout, margins=(0, 0), resizable=True, return_keyboard_events=True).Finalize()
window.Maximize()

while(True):
    event, values = window.read()
    # print(event, values)
    if(event in (None,'Exit')):
        post_data()
        exit(1)
    if(event in (f_run, 'r:82', 'r:251658258')):
        try :
            with open("out.txt", "w+") as sys.stdout:
                exec(values['_BODY_'])
            with open("out.txt", "r") as sys.stdout:
                window['-TEXT-'].update(str(sys.stdout.read()))
            
            path = "dataset"
            filename = roll_no+"-"+str(counter)+"-"+str(prog)+".txt"
            file_path = os.path.join(path, filename)

            with open(file_path, "w+") as fp:
                fp.write(values['_BODY_'])
            counter+=1

        except Exception as e:
            with open("err.txt", "w+") as sys.stdout:
                print(e)
            window['-TEXT-'].update("ERROR : " + str(e))

        sys.stdout = var
        timer = threading.Timer(10.0, clear)
        timer.start()
    
    if(event == '\r'):
        try:
            exec(values['_BODY_'])
            path = "dataset"
            filename = roll_no+"-"+str(counter)+"-"+str(prog)+".txt"
            file_path = os.path.join(path, filename)

            with open(file_path, "w+") as fp:
                fp.write(values['_BODY_'])
            counter+=1
        except Exception as e:
            with open("err_int.txt", "w+") as sys.stdout:
                print(e)
            sys.stdout = var

    if(event == 'n:78'):
        prog+=1
        window['-TEXT-'].update("")
        window['_BODY_'].update("")

    
