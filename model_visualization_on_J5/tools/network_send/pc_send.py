'''pc sender'''
import os
import zmq
import cv2
import argparse
import numpy as np
import time
import test_pb2


def mergeUV(u, v):
    if u.shape == v.shape:
        uv = np.zeros(shape=(u.shape[0], u.shape[1] * 2))
        for i in range(0, u.shape[0]):
            for j in range(0, u.shape[1]):
                uv[i, 2 * j] = u[i, j]
                uv[i, 2 * j + 1] = v[i, j]
        return uv
    else:
        raise ValueError("size of Channel U is different with Channel V")

def rgb2nv12_calc(image):
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b)
    u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
    v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
    uv = mergeUV(u, v)
    yuv = np.vstack((y, uv))
    return yuv.astype(np.uint8)

class BPUBoard:
    def __init__(self, end_point):
        self.task_type = 1
        self.img_info = test_pb2.image_info()
        self.context = zmq.Context()
        # self.socket = self.context.socket(zmq.PUSH)
        # self.socket.setsockopt(zmq.SNDHWM, 2)
        self.end_point = end_point
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 3000)
        self.socket.connect(end_point)

    def __del__(self):
        self.socket.close()
        self.context.destroy()

    def serialize(self, img_name, img_height, img_width, img_valid_height, img_valid_width, modle_name=""):
        self.img_info.image_height = img_height
        self.img_info.image_width = img_width
        self.img_info.image_valid_height = img_valid_height
        self.img_info.image_valid_width = img_valid_width
        self.img_info.image_name = img_name
        self.img_info.task_type = 1
        return self.img_info.SerializeToString()

    def send_msg(self, img, img_height, img_width, img_valid_height, img_valid_width, img_name):
        pb_info = self.serialize(img_name, img_height, img_width, img_valid_height, img_valid_width)
        while True:
            try:
                self.socket.send(pb_info, zmq.NOBLOCK | zmq.SNDMORE)
                self.socket.send(img, zmq.NOBLOCK)
                self.socket.recv()
                break
            except zmq.ZMQError:
                # reset the connect
                self.socket.close()
                self.context.destroy()
                time.sleep(1.5)
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.RCVTIMEO, 3000)
                self.socket.connect(self.end_point)
                continue
        return 0


def send_images(ip, input_file_path, is_loop):
    end_point = "tcp://" + ip + ":6680"
    bpu_board = BPUBoard(end_point)
    files = os.listdir(input_file_path)
    image_count = len(files)
    loop = 1
    if is_loop == "true":
        loop_flag = 0
    else:
        loop_flag = 1
    circle = 0
    while loop:
        for i in range(image_count):
            file_path = os.path.join(input_file_path, files[i])
            img_bgr = cv2.imread(file_path)
            img_height = img_bgr.shape[0]
            img_width = img_bgr.shape[1]
            nv12_array = rgb2nv12_calc(img_bgr)
            bpu_board.send_msg(nv12_array,img_height, img_width, img_height, img_width, files[i])
            print('send %dth message: %s' % (i, files[i]))
        circle += 1
        print("Send %d circle!!!" % (circle))
        time.sleep(1)
        loop = loop - loop_flag
    print("Send over!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_path', type=str, required=True,
                        help='list file which contains name of test images.')
    parser.add_argument('--board_ip', type=str, required=True,
                        help='ip address of board.')
    parser.add_argument('--is_loop', type=str, required=True,
                        help='switch, to define loop send or not, true or false.')
    args = parser.parse_args()

    send_images(args.board_ip, args.input_file_path, args.is_loop)
