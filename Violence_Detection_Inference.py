
import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model


class VideotoData:

    def __init__(self, args):
        self.width = args.width
        self.height = args.height

    def get_data(self, filename):
        '''
        input  : path of avi file
        output : video numpy data in [nframes, 32,32] shape where nframes is the name of
                 frames in the video. The video will be in grayscale
        '''
        cap = cv2.VideoCapture(filename)
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        framearray = []

        for i in range(int(nframes)):
            ret, frame = cap.read()
            print(frame.shape)
            frame = cv2.resize(frame, (self.height, self.width))
            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()

        return np.array(framearray)


class Datato3D:

    def __init__(self, args):

        self.depth = args.depth
        self.model_path = args.model_path
        self.weights_path = args.weights_path

    def videoto3D(self, videoarray):
        '''
        input : video numpy data in [nframes, 32,32]
        output: vid3D file in [10,32,32] format
        '''

        vid3D = []
        nframes = videoarray.shape[0]

        print(f'n_frames: {nframes}')

        if (nframes >= self.depth):
            frames = [x * nframes / self.depth for x in range(self.depth)]
        else:
            frames = [x for x in range(int(nframes))]

        print(f'frames: {len(frames)}, {frames}')

        for i in range(len(frames)):
            vid3D.append(videoarray[int(frames[i]), :, :])

        return np.array(vid3D)

    def inference(self, video_frames):

        category = {0: "No Violence",
                    1: "Violence"}

        c3D_model = load_model(os.path.join(os.getcwd(), self.model_path))
        c3D_model.load_weights(os.path.join(os.getcwd(), self.weights_path))
        vid3D = self.videoto3D(video_frames)
        vid3D = np.expand_dims(vid3D, axis=3)
        vid3D = vid3D.transpose((1, 2, 0, 3))
        vid3D = np.expand_dims(vid3D, axis=0)

        y_pred = c3D_model.predict(vid3D)

        return category[np.argmax(y_pred)]


def parse_args1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)

    return parser.parse_args()


def parse_args2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--model_path", type=str, default='ViolenceDetection_Models/c3d_model.h5')
    parser.add_argument("--weights_path", type=str, default='ViolenceDetection_Models/c3d_model_weights.h5')

    return parser.parse_args()


if __name__ == "__main__":

    args1 = parse_args1()
    # filename = 'fi10.avi'
    filename = '/media/user/New Volume/MTech/ITSS/SG_ViolentVideos/2Men_SerangoonFight.mp4'
    vidData = VideotoData(args1)
    video_frames = vidData.get_data(filename)   # video data in format [nframes, 32,32]
    print(video_frames.shape)

    args2 = parse_args2()
    vid3D = Datato3D(args2)
    action = vid3D.inference(video_frames)
    print(f'RESULT: {action}')
