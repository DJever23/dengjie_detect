import cv2
import sys
import numpy as np
import os.path
 
# 初始化参数
confThreshold = 0.5  # 置信度阈值
nmsThreshold = 0.4  # 非最大抑制阈值
inpWidth = 416 # 网络输入图像的宽度
inpHeight = 416 # 网络输入图像的高度
 
# 加载类名
classesFile = '../data/coco.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
 
# 模型的配置文件和权值文件
modelConfiguration = '../cfg/yolov3.cfg'
modelWeights = '../cfg/yolov3.weights'
 
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL) 
# 获取输出层的名称
def getOutputsNames(net):
    # 获取网络中所有层的名称
    layersNames = net.getLayerNames()
    # 获取输出层的名称
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 
# 绘制预测得到的边界框
def drawPred(classId, conf, left, top, right, bottom):
    # 绘制边界框
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
    
    # 通过类名获取标签值和置信度
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)
        
    # 在边界框上显示标签值
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
 
# 使用非最大值抑制移除低置信度的边界框
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    
    # 扫描从网络输出的所有边界框并仅保留
    # 置信度得分很高的边界框，将框的类标签指定为具有最高分数的类。
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    #print('boxes',boxes)
    #print('confidences',confidences)
    #print('classIds',classIds)
                
    # 执行非最大抑制以消除置信度较低的冗余重叠框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
 
# 处理输入
winName = 'Deep learning object detection in OpenCV'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
outputFile = 'yolo_out_py.avi'
cap = cv2.VideoCapture('../test/test_video/video1.mp4')
 
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
 
while cv2.waitKey(1) < 0:
    # 从视频获取帧
    hasFrame, frame = cap.read()
    
    # 在视频结束时终止程序
    if not hasFrame:
        print('Done processing !!!')
        print('Output file is stored as ', outputFile)
        cv2.waitKey(3000)
        break
    
    # 从框架创建4D blob。
    blob = cv2.dnn.blobFromImage(frame, 1/ 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    
    # 设置网络的输入
    net.setInput(blob)
    
    # 运行前向传递以获得输出层的输出
    outs = net.forward(getOutputsNames(net))
    
    # 移除低置信度的边界框
    postprocess(frame, outs)
    
    # 提出效率信息。 函数getPerfProfile返回推理的总时间（t）和每个层的时间（在layersTimes中）
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    
    vid_writer.write(frame.astype(np.uint8))
    cv2.imshow(winName, frame)
