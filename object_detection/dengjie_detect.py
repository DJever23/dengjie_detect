import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# 增加导入cv包，以及获取摄像头设备号
import cv2

# video = "http://admin:admin@192.168.0.13:8081"
# video = 0
video = "../1.mp4"
cap = cv2.VideoCapture(video)

# 从utils模块引入label_map_util和visualization_utils,label_map_util用于后面获取图像标签和类别，visualization_utils用于可视化。
#from object_detection.utils import label_map_util as label_map
from utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
from utils import visualization_utils as vis_util

# import label_map_util
# import visualization_utils as vis_util

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True  # allocate dynamically

# 添加模型路径：
CWD_PATH = os.getcwd()  # os.getcwd() 方法用于返回当前工作目录。
PATH_TO_CKPT = os.path.join(CWD_PATH, '../ssd_mobilenet_v1_coco_2018_01_28', 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# 加载模型
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)

        tf.import_graph_def(od_graph_def, name='')

# 加载lable map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
i = 0
# 核心代码
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            stime = time.time()  # 计算起始时间
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            print('image_np_expanded', image_np_expanded.shape)
            print('image_np_expanded.ndim', image_np_expanded.ndim)
            if image_np_expanded.ndim != 4:
                cap.release()
                cv2.destroyAllWindows()
                break
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores), category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            print('FPS{:.1f}'.format(1 / (time.time() - stime)))
            cv2.imwrite('../result_frame/result_frame_%d.jpg' % i, image_np)
            i += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
cap.release()
cv2.destroyAllWindows()
out_video = 1
if out_video:
    img = cv2.imread('/home/dengjie/dengjie/project/detection/from_blog/result_frame/result_frame_0.jpg',1)
    image_name = []
    isColor = 1
    fps = 30.0
    frameWidth = img.shape[1]
    frameHeight = img.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../result_video.avi', fourcc, fps, (frameWidth, frameHeight), isColor)
    root = '../result_frame'
    list = os.listdir(root)
    print('list',list)
    print(len(list))
    for i in range(len(list)):
        frame = cv2.imread('/home/dengjie/dengjie/project/detection/from_blog/result_frame/result_frame_%d.jpg'%i,1)
        out.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    print('video has already saved.')