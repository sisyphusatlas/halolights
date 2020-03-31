#get video cam []
#divide fov into 3 segments (a, b, c) [X]
#get person contour [X]
#return segment that contour falls in [X]


import numpy as np
import time

"""
++++++++++++++++++++++++++++LIME+++++++++++++++++++++++++++++++++++
"""
#For dark rooms
#from algorithms import lime





"""
++++++++++++++++++++++++++++LOAD DEPTH DEEP NN+++++++++++++++++++++++++++++++++++
"""

from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')
model = "C:/Users/XENUS/Desktop/pink/program/Projects/halolights scripts/cv/DenseDepth-master/nyu.h5"
# Load model into GPU / CPU
model = load_model(model, custom_objects=custom_objects, compile=False)


"""
++++++++++++++++++++++++++++VID INPUT SET UP+++++++++++++++++++++++++++++++++++
"""
import cv2
camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

#camera = cv2.VideoCapture("videoplayback.mp4")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not camera.isOpened():
    print("Camera is fucked up")


"""
+++++++++++++++++++++++++++++++++CV VARS++++++++++++++++++++++++++++++
"""

#import mobilenetSSD from files
#Densedepth NN
proto = r'C:/Users/XENUS/Desktop/pink/program/Projects/halolights scripts/cv/DenseDepth-master/MobileNetSSD_deploy.prototxt'
weights = r'C:/Users/XENUS/Desktop/pink/program/Projects/halolights scripts/cv/DenseDepth-master/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(proto, weights)
conf_level = 0.5
timelast = time.time()
classNames = { 0: 'background', 5: 'bottle', 8: 'cat', 11: 'diningtable', 
              12: 'dog', 15: 'person', 20: 'tvmonitor' }




"""
+++++++++++++++++++++++++++++++++MQTT CLIENT++++++++++++++++++++++++++++++
"""
import paho.mqtt.client as mqtt
#MQTT FUNCTIONS
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))


client = mqtt.Client(client_id="AI CONTROLLER000")
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(username="lol", password="verygood")
client.connect("192.168.0.18", 1883, 60)


"""
++++++++++++++++++++++++++++++MAIN LOOP+++++++++++++++++++++++++++++++++
"""

midPointX = 0
midPointY = 0
coord012 = None #CAMERA LEFT TO RIGHT COORDINATE
lastcoord012 = 3 
coord102030 = None #DEPTH MAP FRONT TO BACK COORDINATE
lastcoord102030 = 40 #set just beyond
fullcoords = None # coord012 + coord102030
lastfullcoords = 43 
focusedDepthMean = 0

while camera.isOpened():
    #ret returns True if camera is running, frame grabs each frame of the video feed
    ret, frame = camera.read()
    
    
    frame_resized = cv2.resize(frame, (300, 300))
    
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    
    net.setInput(blob)
    
   
    
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]
    
    #DO THIS EVERY SECOND
    if time.time() - timelast >= 1:
        #feed forward the net and get the output; assign to detections
        detections = net.forward()
        
        
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            
            if confidence > conf_level:
                class_id = int(detections[0, 0, i, 1]) #class label
    
                
                # Object locaition
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)
                
                #factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0
                widthFactor = frame.shape[1]/300.0
                
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop = int(widthFactor * xRightTop)
                yRightTop = int(heightFactor * yRightTop)
                
                #cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (120, 20, 120), 2)
                
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    #if person is detected, record their position on the x axis
                    #determine which segment it is in (ABC)
                    if classNames[class_id] == "person":
                        #val is the middle of the detected rectangle
                        midPointX = (( xRightTop - xLeftBottom ) / 2.0 + xLeftBottom)
                        midPointY = (( yRightTop - yLeftBottom) / 2.0 + yLeftBottom)
                        if (midPointX < 700):
                            coord012 = 0
                        elif (midPointX >= 700 and midPointX <= 1220):
                            coord012 = 1
                        elif (midPointX > 1220):
                            coord012 = 2
                        else:
                            coord012 = None
                        
                        
                        resizedFrame = frame.copy()
                        #resize to 640x480 for input to DenseDepth NN
                        resizedFrame = cv2.resize(resizedFrame, (640, 480))
                        
                        #recycle heightFactor and widthFactor for purpose of scaling the midpoint
                        heightFactor = frame.shape[0]/240.0
                        widthFactor = frame.shape[1]/320.0
                        
                        relativeMidPointX = midPointX / widthFactor #relative to resized 
                        relativeMidPointY = midPointY / heightFactor
                        #normalize resizedFrame
                        resizedFrame = np.clip(np.asarray(resizedFrame, dtype=float) / 255, 0, 1)
                        
                        resizedFrame = np.stack(resizedFrame, axis = 0)
                        #                             VThis makes the input size correct for the model
                        crop_frame = resizedFrame[np.newaxis, :, :, :]
                        #get depth map
                        depthImage = predict(model, resizedFrame, batch_size=1)
                        #           V change shape back to something readable by cv2
                        depthImage = depthImage[0]
                        
                        #take a rectangle of depth map around the center point of midpoint          
                        focusedDepth = depthImage[int(relativeMidPointY) - 20:int(relativeMidPointY) + 20,
                                                  int(relativeMidPointX) - 10:int(relativeMidPointX) + 10]
                        #display this rectangle
                        cv2.rectangle(depthImage, (int(relativeMidPointX) - 10, int(relativeMidPointY) + 20), 
                                     (int(relativeMidPointX) + 10, int(relativeMidPointY) - 20), 
                                     (0, 120, 0), 2)
                        focusedDepthMean = focusedDepth.mean()
                        
                        
                        #there is a specific focusedDepthMean threshold for each coord012
                        if (coord012 == 0):
                            if (focusedDepthMean <= 0.12): 
                                coord102030 = 10
                            elif (focusedDepthMean >= 0.155): 
                                coord102030 = 30
                            else: #else its going to be in the middle and therefore the middle coordinate
                                coord102030 = 20
                                
                        elif (coord012 == 1):
                            if (focusedDepthMean <= 0.104): 
                                coord102030 = 10
                            elif (focusedDepthMean >= 0.152): 
                                coord102030 = 30
                            else: 
                                coord102030 = 20
                        
                        elif (coord012 == 2):
                            if (focusedDepthMean <= 0.106): 
                                coord102030 = 10
                            elif (focusedDepthMean >= 0.144): 
                                coord102030 = 30
                            else:
                                coord102030 = 20
                        else:
                            coord102030 = None
                        
                        #you cannot add NoneType
                        if coord102030 != None and coord012 != None:
                            fullcoords = coord102030 + coord012
                        
                        cv2.imshow("depth map", depthImage)

                    """
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    """
                    print(label) #print class and confidence 
        timelast = time.time()
        
        #if person didnt move then 
        if (fullcoords != lastfullcoords):
            client.publish("halolights/changeallcolorsub", "(45), (45), (45)")
        

        """
        COORIDNATE SYSTEM EXPLAINED::


		#this is the 'fullcoords' within the code
        |31|32|22|
        |30|21|12|
        |20|10|11|
        		  *  <--- this is the camera position
        		  so first digit is the depth, second digit is 1 of 3 areas on the camera frame

		#name of lights (_0, _1, _2, _3, ...etc)
        |0|1|2|`
        |3|4|5|
        |6|7|8| 
        	   * <--- camera position
        """

        #client.publish publishes MQTT messages to preset light names (the names are 0-8 as described at the end of the payload (_7, _8... etc))
        if(fullcoords == 10 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_7")
        elif(fullcoords == 20 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_6")
        elif(fullcoords == 30 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_3")
        elif(fullcoords == 11 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_8")
        elif(fullcoords == 21 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_4")
        elif(fullcoords == 31 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_0")
        elif(fullcoords == 12 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_5")
        elif(fullcoords == 22 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_2")
        elif(fullcoords == 32 and fullcoords != lastfullcoords):
            client.publish("halolights/changenumbercolorsub", "(255), (0), (0)_1")
        
        lastfullcoords = fullcoords
        lastcoord012 = coord012
        lastcoord102030 = coord102030
    #Draw segment boundaries
    #1920 / 3 = 640
    cv2.line(frame, (700, 0), (700, 1080), (0, 0, 255), 3)
    cv2.line(frame, (1220, 0), (1220, 1080), (0, 0, 255), 3)
    
    cv2.circle(frame, (int(midPointX), int(midPointY)), 30, (120, 0, 120), -1)
    
    cv2.putText(frame, "person found in coords: {}".format(fullcoords), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 75, 255), 3)
    cv2.putText(frame, "focus depth mean: {}".format(focusedDepthMean), (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 75, 255), 5)
    #render frame    
    cv2.imshow('frame', frame)
    
    
    #break when q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

camera.release()
cv2.destroyAllWindows()

