import cv2
import os
import time
from config import config
import imutils
from imutils import paths
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import expand_dims


# Load the Face Model
face_net = cv2.dnn.readNetFromCaffe(config.PROTOTXT_PATH, config.WEIGHTS_PATH)
# Load the Mask Model
mask_net = load_model(config.MODELV4)

def detectAndPredictMask(frame, face_model, mask_model):
    
    """Function that detects the face, process the image and then assigns the 
        corresponding predicitions to the face.
        
        Args:
            frame (image array) : The images as numpy arrays
            face_model (face_net) : face net loaded from load_models method.
            mask_model (mask_net) : mask net loaded from load_models method.

        Returns:
            None
        """
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the face network to get the face detections
    face_model.setInput(blob)
    face_detections = face_model.forward()
    
    # Get the faces, locations and predictions from the face mask model
    faces = []
    face_locations = []
    predictions = []
    
    # Loop over the face detections
    for i in range(0, face_detections.shape[2]):
        # Extract the confidence associated with the face detection
        confidence = face_detections[0, 0, i, 2]
        # Filter out weak detections to ensure the confidence is greater than the minimum
        if confidence > config.CONF_THRESH:
            # Compute the (x, y) coordinates of the bounding box for the object
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            beginX, beginY, stopX, stopY = box.astype("int")
            # Making sure the bounding boxes fall within the dimensions of the frame
            beginX, beginY = max(0, beginX), max(0, beginY)
            stopX, stopY = min(w - 1, stopX), min(h - 1, stopY)
            try:
                # Slice the frame, convert to RGB and preprocess
                face = frame[beginY:stopY, beginX:stopX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (112, 112))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = expand_dims(face, axis=0) #expand to fit the frame
                faces.append(face)
                face_locations.append((beginX, beginY, stopX, stopY))
                # Only make predictions if atleast one face is detected
                if len(faces) > 0:
                    # For faster processing we will make both predictions(0 & 1) at the same time
                    predictions.append(mask_model.predict(faces))
            except:
                print('Please move your head back a bit!')
    
    return (face_locations, predictions)

# Initialize the webcam
webcam = cv2.VideoCapture(config.VIDEO_PATH2)
time.sleep(2.0)
# Check if camera opened successfully
if (webcam.isOpened() == False):
    print("\nUnable to read camera feed")

while True:
    success, frame = webcam.read()
    # Run when a frame is detected
    if success == True:
        frame = imutils.resize(frame, width=700)
        locs, preds = detectAndPredictMask(frame, face_net, mask_net)
        loc_dict = {}
        pred_dict = {}
        pred_list = []
        # Only make predictions when a face is detected
        if preds != []:
            # for i in range(0, len(locs)):
            #     # Grab the unique face id(i) and append it to their corresponding
            #     # faces(face locations)
            #     loc_dict[i]=locs[i]
            #     for k, v in loc_dict.items():
            #         # Unpack the bounding box for each face
            #         (beginX, beginY, stopX, stopY) = (v[0], v[1], v[2], v[3])
                    
            #         for j in range(0, len(preds)):
            #             # Grab the unique face id(j) and append it to their corresponding
            #             # face predictions
            #             pred_dict[j]=preds[j]
            #             for _, pred_percent in pred_dict.items():
            #                 pred_list.append(pred_percent)
                    # label = np.argmax(preds)
                    # if label == 0:
                    #     cv2.putText(frame, f"Incorrect Mask {(pred_list[0][0][0]*100):.2f}%", (beginX, beginY-8), config.FONT, 0.5, (15, 30, 100), 2)
                    #     cv2.rectangle(frame, (beginX, beginY), (stopX, stopY), (15, 30, 100), 2)
                
                    # elif label == 1:
                    #     cv2.putText(frame, f"Mask {(pred_list[0][0][1]*100):.2f}%", (beginX, beginY-8), config.FONT, 0.5, (0, 255, 15), 2)
                    #     cv2.rectangle(frame, (beginX, beginY), (stopX, stopY), (0, 255, 15), 2)
                        
                    # else:
                    #     cv2.putText(frame, f"No Mask {(pred_list[0][0][2]*100):.2f}%", (beginX, beginY-8), config.FONT, 0.5, (0, 0, 255), 2)
                    #     cv2.rectangle(frame, (beginX, beginY), (stopX, stopY), (0, 0, 255), 2)
            for box, pred in zip(locs, preds):
                (beginX, beginY, stopX, stopY) = box
                (wrongMask, Mask, noMask) = pred[0]
                label = "Incorrect Mask" if wrongMask > Mask and wrongMask > noMask else "Mask" if Mask > wrongMask and Mask > noMask else "No Mask"
                # Include probability in the model
                label = f"{label} {(max(wrongMask, Mask, noMask)*100):.2f}%"
                color = (15, 30, 100) if wrongMask > Mask and wrongMask > noMask else (0, 255, 15) if Mask > wrongMask and Mask > noMask else (0, 0, 255)
                cv2.putText(frame, label, (beginX, beginY-8), config.FONT, 0.5, color, 2)
                cv2.rectangle(frame, (beginX, beginY), (stopX, stopY), color, 2)
                    
        
        cv2.imshow('Real_time', frame)
        # Exit when Esc key is pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

    else:
        break
    
webcam.release()
cv2.destroyAllWindows()