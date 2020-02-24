import cv2
import os
import sys

img = cv2.imread('/home/dengjie/dengjie/project/detection/from_blog/result_frame/result_frame_0.jpg',1)
image_name = []
isColor = 1
fps = 30.0
frameWidth = 1920  # img.shape[1]
frameHeight = 1080  # img.shape[0]
print(img.shape)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../result_video.avi', fourcc, fps, (frameWidth, frameHeight), isColor)
#root = os.path.dirname(__file__)
root = '../result_frame'
list = os.listdir(root)
print('list',list)
print(len(list))
for i in range(len(list)):
    frame = cv2.imread('/home/dengjie/dengjie/project/detection/from_blog/result_frame/result_frame_%d.jpg'%i,1)
    out.write(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
#cap.release()
out.release()
print('video has already saved.')