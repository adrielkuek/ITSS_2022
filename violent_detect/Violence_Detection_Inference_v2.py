import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model


# Default Parameters
c3d_width  = 32
c3d_height = 32
c3d_depth  = 10
model_path   = 'c3d_model.h5'
weights_path = 'c3d_model_weights.h5'

c3d_model = load_model(os.path.join(os.getcwd(), model_path))
c3d_model.load_weights(os.path.join(os.getcwd(), weights_path))

class Datato3D:

    def __init__(self, depth, c3d_model):

        self.depth = depth
        self.c3d_model = c3d_model

    def videoto3D(self, videoarray):
        '''
        input : video numpy data in [nframes, 32,32]
        output: vid3D file in [10,32,32] format
        '''

        vid3D = []
        nframes = videoarray.shape[0]

        if (nframes >= self.depth):
            frames = [x * nframes / self.depth for x in range(self.depth)]
        else:
            frames = [x for x in range(int(nframes))]

        for i in range(len(frames)):
            vid3D.append(videoarray[int(frames[i]), :, :])

        return np.array(vid3D)

    def inference(self, video_frames):

        category = {0: "No Violence",
                    1: "Violence"}

        # c3D_model = load_model(os.path.join(os.getcwd(), self.model_path))
        # c3D_model.load_weights(os.path.join(os.getcwd(), self.weights_path))
        vid3D = self.videoto3D(video_frames)
        vid3D = np.expand_dims(vid3D, axis=3)
        vid3D = vid3D.transpose((1, 2, 0, 3))
        vid3D = np.expand_dims(vid3D, axis=0)

        y_pred = self.c3d_model.predict(vid3D)

        # return category[np.argmax(y_pred)]
        return np.argmax(y_pred)


# Create a deque buffer to store 4 secs worth of frames (120 frames for 30fps) to input into c3d
buffer_len = 120
c3d_videoInput = deque(maxlen=buffer_len)
filename = 'Test_Video.mp4'
#filename = 'E:/RWF-2000/train/no10.avi'
# collect 120 frames for c3d

cap = cv2.VideoCapture(filename)
nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
vid3D = Datato3D(c3d_depth, c3d_model)

for idx in range(int(nframes)):
    ret, img_c3d = cap.read()
    if (idx+1) % buffer_len != 0:
        # resize image frame and convert to grayscale
        img_c3d = cv2.resize(img_c3d, (c3d_height, c3d_width))
        c3d_videoInput.append(cv2.cvtColor(img_c3d, cv2.COLOR_BGR2GRAY))
    else:
        startc3d_time = time.time()
        print(np.array(c3d_videoInput).shape)
        # Convert video segment to 3D input
        curr_action = vid3D.inference(np.array(c3d_videoInput)))
        print(curr_action)
