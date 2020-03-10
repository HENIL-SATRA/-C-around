# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:35:57 2020

@author: HENIL SATRA
"""


import numpy as np
import cv2
from gtts import gTTS

CLASSES_OF_OBJECTS = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]


COLORS_OF_OBJECTS = np.random.uniform(0, 255, size=(len(CLASSES_OF_OBJECTS), 3))
print("[INFORMATION] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
image = cv2.imread("images/example_09.jpg")
(h, w) = image.shape[:2]
cv2.imwrite('D:/hackathon/see_around/html/img/Input.jpg', image)



blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
    (300, 300), 127.5)
print("[INFORMATION] computing object detections for the input image...")
net.setInput(blob)
detections = net.forward()


for i in np.arange(0, detections.shape[2]):
    
    confidence = detections[0, 0, i, 2]
    
    if confidence > 0.2:
        
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        label = "{}: {:.2f}%".format(CLASSES_OF_OBJECTS[idx], confidence * 100)
        print("[Information regarding given image] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY),
            COLORS_OF_OBJECTS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_OF_OBJECTS[idx], 2)
        

print("The object in front of you is :") 
print(label)
cv2.imshow("Output", image)
cv2.imwrite('D:/hackathon/see_around/html/img/Output.jpg', image)
cv2.waitKey(0)
mytext1=("the object in front of you is : ")
mytext2=label
mytext= mytext1 + mytext2
language = 'en'
myobject = gTTS(text=mytext, lang=language)
myobject.save("information.mp3")    


