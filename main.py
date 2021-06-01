import cv2, os, random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import streamlit as st
# import h5py
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

IMG_WIDTH, IMG_HEIGHT = 416, 416
model = tf.keras.models.load_model("./model/face_detect_model.h5")
MODEL = './yolo/yolov3-face.cfg'
WEIGHT = './yolo/yolov3-wider_16000.weights'
img_folder_path = './image'

# 3 line below is for running on streamlit
# st.title("Webcam Live Feed")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Check camera
if not cap.isOpened():
    raise IOError("Cannot open webcam")

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)

# Use this 2 line if your OpenCV run on GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Use this 2 line if your OpenCV run on CPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Auto labeling
folder_list = os.listdir(img_folder_path)
folder_list.sort()
labels = {}
for i in range(len(folder_list)):
    labels[i] = folder_list[i]

count = 0

while True:
    ret, frame = cap.read()
    # frame is now the image capture by the webcam (one frame of the video)
    
    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0,0,0], 1, crop=False)
    # Set model input
    net.setInput(blob)
    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()
    # Run 'prediction'
    outs = net.forward(output_layers)
    

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    # Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores. Assign the box's class label as the class with the highest score.

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
        # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                # Find the top left point of the bounding box
                topleft_x = int(center_x - width/2)
                topleft_y = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    result = frame.copy()
    final_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        # Draw bouding box with the above measurements
        cv2.rectangle(result, (left, top), (left + width, top + height), (0,255,0), 2)
        try:
            predict_img = frame[top: top+height, left: left + width]
            
            predict_img = cv2.resize(predict_img, (150, 150), cv2.INTER_AREA)
            img_array = img_to_array(predict_img)
            img_array = np.expand_dims(img_array, axis=0)
            predict_img_processed = preprocess_input(img_array)

            predictions = model.predict(predict_img_processed)
            print(predictions)
            if max(predictions[0]) >= 0.75:
                text = f'{labels[np.argmax(predictions[0])]}_{confidences[i]:.2f}'
                cv2.putText(result, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            else:
                text = f'Unknown_{confidences[i]:.2f}'
                cv2.putText(result, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        except:
          pass
    text2 = f'Number of faces detected: {len(indices)}'
    cv2.putText(result, text2, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    
    cv2.imshow('face detection', result)

    # 2 line below is for running on streamlit
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
    # FRAME_WINDOW.image(result)

    c = cv2.waitKey(1)
    # Break when pressing ESC
    if c == 27:
        break
    # Capture image of your face when press c
    elif c == ord('c'):
        for i in indices:
            i = i[0]
            box = boxes[i]
            final_boxes.append(box)

            # Extract position data
            left, top, width, height = box[0], box[1], box[2], box[3]
            face = frame[top: top+height, left: left + width]
            cv2.imwrite('C:/Users/Dang Quang/Facial_Recognition/image/quang/image'+str(count)+'.jpg', face)
            count +=1

cap.release()
cv2.destroyAllWindows()