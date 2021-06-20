import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import os

#return frames
class DenseFlowByFrame:
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

    def get_flow_by_frame(self):
        if self.frame_iter == self.length:
            print("No More Frames")
            return

        print(str(round((self.frame_iter/self.length)*100, 2))+"%")
        ret, frame = self.cap.read()
        if frame is None:
            return
        frame = cv.resize(frame, dsize=(224,224))
        frame_gpu = cv.cuda_GpuMat()
        frame_gpu.upload(frame)

        # use gpu
        gray = cv.cuda.cvtColor(frame_gpu, cv.COLOR_BGR2GRAY)
        gpu_flow = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0,)
        # Calculates dense optical flow by Farneback method
        # use gpu
        gpu_flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow, self.prev_gray, gray, None,)
        # 아래 과정에서, GPU-> CPU-> GPU -> CPU로 텐서가 이동한다. GPU에서 바로 넘길 수 있으면 좋겠으나 방법을 찾지 못했다.
        flow = gpu_flow.download()
        torch_flow = torch.tensor(flow, device=self.cuda)

        torch_flow = torch_flow/12+0.5  #10% of height

        # 0, 1사이로 고정
        torch_flow = torch.clamp(torch_flow, 0, 1)

        #padding
        m = torch.nn.ZeroPad2d((0, 1, 0, 0))
        torch_flow = m(torch_flow)

        torch_flow = torch_flow.cpu()
        npy_flow = torch_flow.numpy()        

        self.prev_gray = gray
        self.frame_iter += 1

        return npy_flow
        
    def set_video(self, video_path):
        self.video_path = video_path
        if self.is_video is True:
            self.cap.release()
        self.cap = cv.VideoCapture(self.video_path)
        self.length = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1
        self.cuda = torch.device('cuda')

        ret, first_frame = self.cap.read()
        first_frame = cv.resize(first_frame, dsize=(224,224))

        if ret is False:
            print("ERROR: video loading failed", video_path)
            return

        first_frame_gpu = cv.cuda_GpuMat()
        # upload on gpu
        first_frame_gpu.upload(first_frame)
        first_frame_gpu = cv.cuda.pyrDown(first_frame_gpu)

        # use gpu
        self.prev_gray = cv.cuda.cvtColor(first_frame_gpu, cv.COLOR_BGR2GRAY)
        self.frame_iter = 0
        self.is_video = True