import numpy as np
import os
import sys
import tensorflow as tf
import time
from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
# 增加导入cv包，以及获取摄像头设备号
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取文件的绝对路径，再获取文件目录
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, './utils'))
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
import label_map_util
import visualization_utils as vis_util

# 从utils模块引入label_map_util和visualization_utils,label_map_util用于后面获取图像标签和类别，
# visualization_utils用于可视化。

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True  # allocate dynamically


def mode_select(state):
    if state not in {'picture', 'video','real_time'}:
        raise ValueError('{} is not a valid argument!'.format(state))
    if state == 'picture':
        return 1
    elif state == 'video':
        video = "../video5.mp4"
    else:
        video = "http://admin:admin@192.168.0.13:8081"
        # video = 0
    cap = cv2.VideoCapture(video)
    return cap


def load_model():
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
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print('models have already loaded.')
    return detection_graph, category_index


def detection(image_np, cap):
    stime = time.time()  # 计算起始时间
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # print('image_np_expanded', image_np_expanded.shape)
    # print('image_np_expanded.ndim', image_np_expanded.ndim)

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
    fps = 1 / (time.time() - stime)
    process_time = time.time() - stime
    if cap == 1:
        return image_np, process_time
    else:
        return image_np, fps


def video(out_video=True):
    if out_video:
        img = cv2.imread('/home/dengjie/dengjie/project/detection/from_blog/result_frame/result_frame_0.jpg', 1)
        isColor = 1
        FPS = 20.0
        frameWidth = img.shape[1]
        frameHeight = img.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('../result_video.avi', fourcc, FPS, (frameWidth, frameHeight), isColor)
        root = '../result_frame'
        list = os.listdir(root)
        print(len(list))
        for i in range(len(list)):
            frame = cv2.imread(
                '/home/dengjie/dengjie/project/detection/from_blog/result_frame/result_frame_%d.jpg' % i, 1)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.release()
        print('video has already saved.')
        return 1
    else:
        return 0


def test_pic(path, cap):
    if cap == 1:
        test_list = os.listdir(path)
        for i in range(len(test_list)):
            image_np = cv2.imread('/home/dengjie/dengjie/project/detection/from_blog/test_pic/' + test_list[i], 1)
            image, process_time = detection(image_np, cap)
            cv2.imwrite('../result_test/result_pic_%d.jpg' % i, image)
            print('spend{:.5f}s'.format(process_time))
            cv2.waitKey()
        cv2.destroyAllWindows()
        return 1


if __name__ == '__main__':
    detection_graph, category_index = load_model()  # 加载模型
    path = '../test_pic'  # 待检测图片的路径
    i = 0  # 图片计数
    state = 'picture'  # 检测模式选择,state = 'video','picture','real_time'
    with detection_graph.as_default():
        with tf.Session(config=config) as sess:
            cap = mode_select(state)
            if cap == 1:
                flag = test_pic(path, cap)
                if flag:
                    print('Detect Successfully!')
            else:
                while True:
                    ret, image_np = cap.read()
                    '''
                    cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，
                    如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
                    '''
                    if ret == 0:
                        cap.release()
                        cv2.destroyAllWindows()
                        print('video has already finished.')
                        break

                    image, fps = detection(image_np, cap)

                    if state == 'video':
                        cv2.imwrite('../result_frame/result_frame_%d.jpg' % i, image)
                        i += 1
                        print('FPS{:.1f}'.format(fps))
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break
                    else:
                        print('FPS{:.1f}'.format(fps))
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            print('Real-Time Detection has finished!')
                            cv2.destroyAllWindows()
                            break
                cap.release()
                cv2.destroyAllWindows()
                if state == 'video':
                    val = video(out_video=True)
                    if val:
                        print('Detect Successfully!')
                    else:
                        print('Unsaved video.')
