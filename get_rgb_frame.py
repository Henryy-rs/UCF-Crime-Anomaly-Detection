import cv2 as cv
import numpy as np
import os

class RGBFrame:
    def __init__(self, video_path=""):
        if video_path != "":
            self.set_video(video_path)
        else:
            self.video_path = video_path
            self.frame_iter = 0
            self.length = 0
            self.is_video = False
    def __len__(self):
        return self.length
        
    def get_rgb_frame(self):
        if self.frame_iter == self.length:
            print("No More Frames")
            return
        ret, frame = self.cap.read()
        if frame is None:
            return
        return np.array(cv.resize(frame, dsize=(224,224)))
     
    def set_video(self, video_path):
        self.video_path = video_path
        if self.is_video is True:
            self.cap.release()
        self.cap = cv.VideoCapture(self.video_path)
        self.length = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        