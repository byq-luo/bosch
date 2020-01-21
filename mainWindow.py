from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import gui
import pathlib
import PIL.Image, PIL.ImageTk
import time
import cv2
import math



class Root(Tk):
    def __init__(self):
        """creates the main window"""
        super(Root, self).__init__()
        self.title("Main Window")
        self.minsize(600, 400)
        self.filename = None
        self.vid = None
        self.delay = 20
        self.wm_iconbitmap('icon2.ico')     # application icon
        self.time = int(round(time.time() * 1000))      # time in ms will be used to playback at the correct speed

        """The frame for file button and file path"""
        self.FileFrame = ttk.LabelFrame(self, text="Open File")
        self.FileFrame.grid(column=0, row=1, padx=20, pady=20)
        self.buttonFile()

        """The frame for the play button"""
        self.VideoFrame = ttk.LabelFrame(self, text="Video")
        self.VideoFrame.grid(column=1, row=1, padx=20, pady=20)
        self.buttonVideo()

        """The video widget"""
        self.videoLabel = ttk.Label(self, anchor=S, image=None)
        self.videoLabel.grid(column=1, row=2,)

    def buttonFile(self):
        self.filebutton = ttk.Button(self.FileFrame, text="Browse A File", command=self.fileDialog)
        self.filebutton.grid(column=1, row=1)

    def buttonVideo(self):
        self.videobutton = ttk.Button(self.VideoFrame, text="Play File", command=self.makeVideo)
        self.videobutton.grid(column=1, row=1)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir=pathlib.Path().absolute(), title="Select A File", filetype=
        (("avi files", "*.avi"), ("all files", "*.*")))
        self.label = ttk.Label(self.FileFrame, text="")
        self.label.grid(column=1, row=2)
        self.label.configure(text=self.filename)
        # App(Tk(), "Tkinter and OpenCV", self.filename)

    def makeVideo(self):
        self.vid = gui.Video(self.filename)
        self.delay = int(1000/self.vid.get_fps())
        self.update()

    def draw_bounding_box(self, uptime, frame):
        # draws a bounding box
        x = int(100*(.75 + .25*math.cos(uptime/200)))
        y = int(100*(.75 + .25*math.sin(uptime/200)))
        w = 100
        h = 100
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def update(self):
        self.time = int(round(time.time() * 1000))
        uptime = self.time + self.delay     # the time that the next frame should be pulled
        if self.vid is not None:
            ret, frame = self.vid.get_frame()
            if frame is None:
                """video has played all the way through"""
                return

            self.draw_bounding_box(uptime, frame)

            if ret:
                photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.videoLabel.photo = photo
                self.videoLabel.config(image=photo)
        self.time = int(round(time.time() * 1000))
        self.after(uptime - self.time, self.update)    # call the function again after the difference in time has passed


root = Root()
root.mainloop()
