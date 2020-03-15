import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import core.utils as utils
from core.config import cfg
import random
import datetime
import argparse
import json
from socket import *


def video_without_saving(ip, threshold):

    # 对方socket
    des_socket = socket(AF_INET, SOCK_STREAM)
    # 链接服务器
    des_socket.connect(('127.0.0.1', 8000))

    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    return_elements = [
        "input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0",
        "pred_lbbox/concat_2:0"
    ]
    pb_file = "./yolov3_coco.pb"
    video_path = ip

    input_size = 416
    graph = tf.Graph()
    return_tensors = utils.read_pb_return_tensors(graph, pb_file,
                                                  return_elements)

    with tf.Session(graph=graph) as sess:
        messageId = 0
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        # 之后修改
        # vid = cv2.VideoCapture(video_path)
        vid = cv2.VideoCapture(0)
        while True:
            # time.sleep(0.01)
            curr_time = datetime.datetime.now()
            timestamp = '%s-%s-%s %s:%s:%s' % (
                curr_time.year, curr_time.month, curr_time.day, curr_time.hour,
                curr_time.minute, curr_time.second)

            return_value, frame = vid.read()

            result_, imgencode = cv2.imencode('.jpg', frame, encode_param)
            data = np.array(imgencode)
            stringData = data.tostring()
            length = len(stringData)


            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                raise ValueError("No image!")
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame),
                                                [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

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
                                             threshold)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = utils.draw_bbox(frame, bboxes)

            result = np.asarray(image)

            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            messageId += 1
            mess_send(des_socket, bboxes, timestamp, messageId, ip, length,stringData)


def mess_send(des_socket, bboxes, timestamp, messageId, ip, length,
              stringData):
    all_boxes = []
    for box in bboxes:
        _ = box.tolist()
        all_boxes.append(_)


    response_data = {
        "messageName": "FuncControlResponse",
        "messageId": messageId,
        "cameraIP": ip,
        "function": "objectDetection",
        "responseStatus": 1,
        "resultData": {
            "timestamp": timestamp,
            "position": all_boxes,
            "image": stringData,
            "Messages": length
        }
    }
    response_data = json.dumps(response_data)
    des_socket.send(response_data.encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameraIP", type=str)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    video_without_saving(args.cameraIP, args.threshold)