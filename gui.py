import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time

class App:
    def __init__(self, window, window_title, video_source):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.vid = Video(video_source)

        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S"))

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(1, self.update)


class Video:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


    def get_frame(self):
        ret = False
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)

        else:
            return(ret, None)


#App(tkinter.Tk(), "Tkinter and OpenCV", "flame.avi")

def main():
    #App(tkinter.Tk(), "Tkinter and OpenCV", "flame.avi")
    print("ran")

