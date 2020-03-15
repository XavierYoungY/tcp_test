import tensorflow as tf
from core.yolov3 import YOLOV3
from core.config import cfg
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def create_pb(ckpt_file):

    pb_file = "./yolov3_coco.pb"


    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    model = YOLOV3(input_data, trainable=False)
    print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

    sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                input_graph_def  = sess.graph.as_graph_def(),
                                output_node_names = output_node_names)

    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())

if __name__ == "__main__":
    ckpt_file = 'checkpoint/yolov3_coco_demo.ckpt'
    create_pb(ckpt_file)
