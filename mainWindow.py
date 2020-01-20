from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import gui
import pathlib
import PIL.Image, PIL.ImageTk



class Root(Tk):
    def __init__(self):
        # creates the main window
        super(Root, self).__init__()
        self.title("Main Window")
        self.minsize(640, 400)
        self.filename = None
        self.vid = None
        self.wm_iconbitmap('icon2.ico')
        # The frame for file button and file path
        self.FileFrame = ttk.LabelFrame(self, text="Open File")
        self.FileFrame.grid(column=0, row=1, padx=20, pady=20)
        self.buttonFile()

        # The frame for the play button
        self.VideoFrame = ttk.LabelFrame(self, text="Video")
        self.VideoFrame.grid(column=1, row=1, padx=20, pady=20)
        self.buttonVideo()
        #self.appcanvas = Canvas(self, width=500, height=500)


        #The video widget
        self.videoLabel = ttk.Label(self, anchor=S, image=None)
        self.videoLabel.grid(column=1, row=2,)

        #
        self.delay = 15
        #self.update()


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
        #gui.App(Tk(), "Tkinter and OpenCV", self.filename)
        self.vid = gui.Video(self.filename)
        self.update()

    def update(self):
        if self.vid is not None:
            ret, frame = self.vid.get_frame()
            if frame is None:
                # video has played all the way through
                return

            if ret:
                photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.videoLabel.photo = photo
                self.videoLabel.config(image=photo)



        self.after(self.delay, self.update)





root = Root()
root.mainloop()
1+1