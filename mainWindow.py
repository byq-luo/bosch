from tkinter import *
#from tkinter import ttk
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
        self.videoLabel = None
        self.fileLabel = None
        self.playButtonLabel = None
        self.pauseButtonLabel = None
        self.vid = None
        self.delay = 20
        self.wm_iconbitmap('icon2.ico')     # application icon
        self.time = int(round(time.time() * 1000))      # time in ms will be used to playback at the correct speed

        """The frame for file button and file path"""
        self.FileFrame = LabelFrame(self, text="Open File")
        self.FileFrame.grid(column=0, row=1, sticky=N, padx=15, pady=15)
        self.buttonFile()

        """The frame for the play button"""
        self.VideoFrame = LabelFrame(self, text="Video")
        self.VideoFrame.grid(column=0, row=2, sticky=N, padx=15, pady=15)
        self.buttonVideo()


    def buttonFile(self):
        self.filebutton = Button(self.FileFrame, text="Browse A File", command=self.fileDialog)
        self.filebutton.grid(column=1, row=1)

    def buttonVideo(self):
        self.videobutton = Button(self.VideoFrame, text="Play File", command=self.makeVideo)
        self.videobutton.grid(column=1, row=1)

    def buttonVideoPlay(self):
        """Destroy the pause button"""
        if self.pauseButtonLabel is not None:
            self.pauseButtonLabel.destroy()
            self.pauseButtonLabel = None

        """The play button label"""
        self.playButtonLabel = Label(self, text=None)
        self.playButtonLabel.grid(row=4, column=0)

        #photo = PhotoImage(file = r"/Users/adamschroth/PycharmProjects/bosch/play.png")
        self.videoPlayButton = Button(self.playButtonLabel, text = "Play", command=self.playVideo)
        self.videoPlayButton.grid(column=1, row=1)

    def buttonVideoPause(self):
        """Destroy the play button"""
        if self.playButtonLabel is not None:
            self.playButtonLabel.destroy()
            self.playButtonLabel = None

        """The pause button label"""
        self.pauseButtonLabel = Label(self, text=None)
        self.pauseButtonLabel.grid(row=4, column=0)

        self.videoPauseButton = Button(self.pauseButtonLabel, text = "Pause", command=self.pauseVideo)
        self.videoPauseButton.grid(column=1, row=1)


    def buttonBackToSelect(self):
        self.backButtonLabel = Label(self, text=None)
        self.backButtonLabel.grid(column=0, row=1)

        self.backButton = Button(self.backButtonLabel, text = "Select another file", command=self.backToSelect)
        self.backButton.grid(column=1, row=1)


    def fileDialog(self):
        filename = filedialog.askopenfilename(initialdir=pathlib.Path().absolute(), title="Select A File", filetypes=
        (("avi files", "*.avi"), ("all files", "*.*")))
        if filename is not '':
            """only sets self.filename and fileLabel if a file was selected"""
            if self.fileLabel is not None:
                self.fileLabel.destroy()
            self.filename = filename
            self.fileLabel = Label(self.FileFrame, text="")
            self.fileLabel.grid(column=1, row=2)
            self.fileLabel.configure(text=self.filename)


    def makeVideo(self):

        self.buttonBackToSelect()

        """The video widget"""
        self.playerLabel = Label(self, image=None)
        self.playerLabel.grid(row=2, column=0, rowspan=2, sticky=N+E+S+W)

        if self.FileFrame is not None:
            self.FileFrame.destroy()
            self.FileFrame = None

        if self.VideoFrame is not None:
            self.VideoFrame.destroy()
            self.VideoFrame = None

        self.buttonVideoPlay()

        self.isPaused = True
        if self.filename is not None:   # a file was selected
            if self.videoLabel is not None:     # get rid of no file selected label
                self.videoLabel.destroy()
                self.videoLabel = None

            self.vid = gui.Video(self.filename)
            self.delay = int(1000/self.vid.get_fps())
            self.update()

        else:   # a file was not selected
            """create label to notify user that there is no file selected"""
            self.videoLabel = Label(self.VideoFrame, text="")
            self.videoLabel.grid(column=0, row=2)
            self.videoLabel.configure(text="No file selected")


    def playVideo(self):
        self.isPaused = False
        self.buttonVideoPause()
        self.update()


    def pauseVideo(self):
        self.isPaused = True
        self.buttonVideoPlay()
        self.update()

    def backToSelect(self):
        """Destroy the video player widget"""
        if self.playerLabel is not None:
            self.playerLabel.destroy()
            self.playerLabel = None

        """Destroy the play button"""
        if self.playButtonLabel is not None:
            self.playButtonLabel.destroy()
            self.playButtonLabel = None

        """Destroy the pause button"""
        if self.pauseButtonLabel is not None:
            self.pauseButtonLabel.destroy()
            self.pauseButtonLabel = None

        """Destroy the select another file label"""
        if self.backButtonLabel is not None:
            self.backButtonLabel.destroy()
            self.backButtonLabel = None

        """The frame for file button and file path"""
        self.FileFrame = LabelFrame(self, text="Open File")
        self.FileFrame.grid(column=0, row=1, sticky=N, padx=15, pady=15)
        self.buttonFile()

        """The frame for the play button"""
        self.VideoFrame = LabelFrame(self, text="Video")
        self.VideoFrame.grid(column=0, row=2, sticky=N, padx=15, pady=15)
        self.buttonVideo()

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
                self.playerLabel.photo = photo
                self.playerLabel.config(image=photo)    # updates the player label to show most current image
        self.time = int(round(time.time() * 1000))
        if not self.isPaused:
            self.after(uptime - self.time, self.update)    # call the function again after the difference in time has passed


root = Root()
root.mainloop()
