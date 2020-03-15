import cv2
import time
import numpy as np
from PIL import Image
import os
import socket
from timeit import time
import sys
import queue
from threading import Thread
import subprocess as sp
q = queue.LifoQueue()
images = queue.LifoQueue()

# RTSP = 'rtsp://admin:123456789bit@10.110.0.14:554/11'
RTSP = 0



def Yolo():
    cap = cv2.VideoCapture(RTSP)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':

    Yolo()
