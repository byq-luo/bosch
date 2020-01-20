from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import gui


class Root(Tk):
    def __init__(self):
        # creates the main window
        super(Root, self).__init__()
        self.title("Main Window")
        self.minsize(640, 400)
        self.filename = None
        self.wm_iconbitmap('icon2.ico')
        # The frame for file button and file path
        self.FileFrame = ttk.LabelFrame(self, text="Open File")
        self.FileFrame.grid(column=0, row=1, padx=20, pady=20)
        self.buttonFile()

        # The frame for the play button
        self.VideoFrame = ttk.LabelFrame(self, text="Video")
        self.VideoFrame.grid(column=1, row=1, padx=20, pady=20)
        self.buttonVideo()

    def buttonFile(self):
        self.filebutton = ttk.Button(self.FileFrame, text="Browse A File", command=self.fileDialog)
        self.filebutton.grid(column=1, row=1)

    def buttonVideo(self):
        self.videobutton = ttk.Button(self.VideoFrame, text="Play File", command=self.makeVideo)
        self.videobutton.grid(column=1, row=1)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=
        (("avi files", "*.avi"), ("all files", "*.*")))
        self.label = ttk.Label(self.FileFrame, text="")
        self.label.grid(column=1, row=2)
        self.label.configure(text=self.filename)
        # App(Tk(), "Tkinter and OpenCV", self.filename)

    def makeVideo(self):
        gui.App(Tk(), "Tkinter and OpenCV", self.filename)





root = Root()
root.mainloop()