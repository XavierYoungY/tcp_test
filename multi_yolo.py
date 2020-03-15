import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import core.utils as utils
from core.config import cfg
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import socket
from timeit import time
import sys
import queue
from threading import Thread
import subprocess as sp
q = queue.LifoQueue()
images = queue.LifoQueue()


def Receive():

    # time.sleep(10)
    cap = cv2.VideoCapture('rtsp://admin:123456789bit@10.110.0.37:554/11')
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Yolo():

    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    return_elements = [
        "input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0",
        "pred_lbbox/concat_2:0"
    ]
    pb_file = "./yolov3_coco.pb"
    input_size = 416
    graph = tf.Graph()
    return_tensors = utils.read_pb_return_tensors(graph, pb_file,
                                                  return_elements)

    with tf.Session(graph=graph) as sess:
        cap = cv2.VideoCapture('rtsp://admin:123456789bit@10.110.0.37:554/12')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = -1
        # RTMP
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ip = '10.110.0.233'
        rtmpUrl = 'rtmp://' + ip + ':8888/live/obj_detection'

        rtmp_init_flag = True
        count = 0
        while True:
            if q.empty() != True:
                frame = q.get()
                q.queue.clear()
                count += 1
                t1 = time.time()
                if frame is None:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                continue

            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame),
                                                [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={return_tensors[0]: image_data})

            pred_bbox = np.concatenate([
                np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                np.reshape(pred_lbbox, (-1, 5 + num_classes))
            ],
                                       axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size,
                                             0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = utils.draw_bbox(frame, bboxes)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" % (1000 * exec_time)
            # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            speed = 3
            # for i in range(speed):
            #     if not images.full():
            #         images.put(result)
            #     else:
            #         print('images fulllllllllllll')

            fps = (fps + (1. / (time.time() - t1))) / 2
            rtmp_fps = int(fps) * speed - 4
            # print(rtmp_fps)
            if rtmp_init_flag:
                proc, rtmp_init_flag = rtmp_init(size, 27, rtmpUrl)
            if not rtmp_init_flag:
                for i in range(speed):
                    # if images.empty() != True:
                    #     img = images.get()
                    rtmp(proc, result)
                    # else:
                    #     print('image empty')

            if count > 1e7: count = 1e7


def rtmp_start():
    cap = cv2.VideoCapture('rtsp://admin:123456789bit@10.110.0.37:554/10')

    # RTMP
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ip = '10.110.0.233'
    rtmpUrl = 'rtmp://' + ip + ':8888/live/obj_detection'
    proc, rtmp_init_flag = rtmp_init(size, 30, rtmpUrl)
    while True:
        if images.empty() != True:
            img = images.get()
            for i in range(2):
                rtmp(proc, img)
        else:
            continue


def rtmp_init(size, fps, rtmpUrl):
    sizeStr = str(size[0]) + 'x' + str(size[1])
    fps = int(fps)
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt',
        'bgr24', '-s', sizeStr, '-r',
        str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast', '-f', 'flv', rtmpUrl
    ]
    proc = sp.Popen(command, stdin=sp.PIPE, shell=False)
    return proc, False


def rtmp(proc, img):
    proc.stdin.write(img.tostring())
    proc.stdin.flush()


if __name__ == '__main__':
    t1 = Thread(target=Receive)

    t1.setDaemon(True)

    t1.start()

    while True:
        Yolo()
