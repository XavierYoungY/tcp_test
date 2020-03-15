import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3
from mAP import main
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np


def get_tensors():
    return_elements = [
        "input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0",
        "pred_lbbox/concat_2:0"
    ]
    pb_file = "./yolov3_coco.pb"
    graph = tf.Graph()
    return_tensors = utils.read_pb_return_tensors(graph, pb_file,
                                                  return_elements)
    return return_tensors, graph


class YoloTest(object):
    def __init__(self, name):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path = cfg.TEST.ANNOT_PATH.rstrip(
            '.txt') + name + '.txt'
        self.weight_file = cfg.TEST.WEIGHT_FILE
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32,
                                             name='input_data')
            self.trainable = tf.placeholder(dtype=tf.bool, name='trainable')

        # model = YOLOV3(self.input_data, self.trainable)
        # self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox
        self.return_tensors, self.graph = get_tensors()

        # with tf.name_scope('ema'):
        #     ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess = tf.Session(
            graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        # self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        # self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image,
                                            [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [
                self.return_tensors[1], self.return_tensors[2],
                self.return_tensors[3]
            ],
            feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([
            np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
            np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
            np.reshape(pred_lbbox, (-1, 5 + self.num_classes))
        ],
                                   axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w),
                                         self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def evaluate(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path):
            shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path):
            shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path):
            shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                if annotation == []:
                    continue
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                bbox_data_gt = np.array(
                    [list(map(int, box.split(','))) for box in annotation[1:]])

                bbox_data_gt = bbox_data_gt.reshape(-1, 5)
                if len(bbox_data_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :
                                                         4], bbox_data_gt[:, 4]

                ground_truth_path = os.path.join(ground_truth_dir_path,
                                                 str(num) + '.txt')

                #print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join(
                            [class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        #print('\t' + str(bbox_mess).strip())
                #print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path,
                                                   str(num) + '.txt')
                bboxes_pr = self.predict(image)

                if self.write_image:
                    image = utils.draw_bbox(image,
                                            bboxes_pr,
                                            show_label=self.show_label)
                    cv2.imwrite(self.write_image_path + image_name, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join(
                            [class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        #print('\t' + str(bbox_mess).strip())


if __name__ == '__main__':

    names = ['_small', '_medium', '_large']
    for name in names:
        Test = YoloTest(name)
        Test.evaluate()
        mAP = []
        ap_dictionary = []
        for i in np.arange(0.5, 1, 0.05):
            map_, ap = main.map(i, draw_plot=False)
            mAP.append(map_)
            ap_dictionary.append(ap)
            path = os.getcwd().rstrip('/mAP')
            os.chdir(path)

        coco_map = sum(mAP) / len(mAP)
        print(coco_map)
        mAP.append(coco_map)
        with open(name+'map.txt', 'w') as f:
            for line in mAP:
                f.write(str(line) + '\n')

        classes = utils.read_class_names(cfg.YOLO.CLASSES)
        ap = {}
        for classname in classes.values():
            for i in ap_dictionary:
                if classname not in ap:
                    if classname not in i:
                        ap[classname]=0.0
                    else:
                        ap[classname] = i[classname]
                else:
                    if classname not in i:
                        ap[classname] +=0.0
                    else:
                        ap[classname] += i[classname]
            ap[classname] = ap[classname] / len(ap_dictionary)
        with open(name + 'ap-classes.txt', 'w') as f:
            for key in ap:
                f.write(str(key) + ': ' + str(ap[key]) + '\n')
