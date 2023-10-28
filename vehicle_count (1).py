# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker1 import *

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('video.mp4')
input_size = 320

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position


# Store Coco Names in a list
classesFile = "/home/201112223/yolo_v3/coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = '/home/201112223/yolo_v3/yolov3-320.cfg'
modelWeigheights = '/home/201112223/yolo_v3/yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_list = []
counting_res = [0, 0, 0, 0]
op_classes = ['car', 'bus', 'truck', 'motorcycle']
line1 = 85
line2 = 90
vertical_lim1 = 135
vertical_lim2 = 220

# Function for count vehicle
def count_vehicle(box_id, image):

    global temp_list
    global counting_res
    x, y, w, h, id, index = box_id
    center = find_center(x, y, w, h)
    ix, iy = center
    if iy > line1 and iy < line2 and ix > vertical_lim1 and ix < vertical_lim2:
        if id not in temp_list:
            temp_list.append(id)
            counting_res[index] = counting_res[index]+1
            cv2.circle(image, center, 2, (255, 0, 0), 2)
        
    return image

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
             
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score 
        cv2.putText(img, name, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 0,
                    lineType=cv2.LINE_AA)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

        if x > vertical_lim1 and name in op_classes:
            if y > line1 and y < line2:
                detection.append([x, y, w, h, op_classes.index(name)])
            elif y+h > line1 and y+h < line2:
                detection.append([x, y, w, h, op_classes.index(name)])
            elif y < line1 and y+h > line2:
                detection.append([x, y, w, h, op_classes.index(name)])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def from_static_image(image):
    img = image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
# Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
# Feed data to the network
    outputs = net.forward(outputNames)
    # Find the objects from the network output
    postProcess(outputs,img)
    # count the frequency of detected classes
    font_size = 0.3
    font_color = (88, 62, 34)
    font_thickness = 1
    cv2.putText(img, "Car:         "+str(counting_res[0]), (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus:         "+str(counting_res[1]), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck:       "+str(counting_res[2]), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorcycle:  "+str(counting_res[3]), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    return img

    cv2.waitKey(0)

cap = cv2.VideoCapture('/home/201112223/Minor project/qmul_junction.avi')
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('/home/201112223/Minor project/output.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'), 30,
                      (frame_width, frame_height))

frames = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        img = from_static_image(frame)
        cv2.line(img, (135, 85), (220, 85), (0, 0, 255), 1)
        cv2.line(img, (135, 90), (220, 90), (0, 0, 255), 1)
        out.write(img)
    else:
        break
