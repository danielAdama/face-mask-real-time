import cv2
import os
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
mask_net = load_model(config.MODELV3)

images = list(paths.list_images(config.IMAGE_PATH))
for image in images:
    image_name = image.split(os.path.sep)[-1].split('.')[0]
    image_format = image.split(os.path.sep)[-1].split('.')[1]
    print(f"Processing Image: '{image_name}' with format {image_format}")
    img = cv2.imread(image)
    
    # Grab the dimensions of an image and then construct a blob
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Pass the blob through the face network to get the face detections
    face_net.setInput(blob)
    face_detections = face_net.forward()
    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        # Filter out weak detections to ensure the confidence is greater than the minimum
        if confidence > config.CONF_THRESH:
            # Compute the (x, y) coordinates of the bounding box for the object
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            beginX, beginY, stopX, stopY = box.astype('int')
            # Making sure the bounding boxes fall within the dimension of the image
            beginX, beginY = max(0, beginX), max(0, beginY)
            stopX, stopY = min(w - 1, stopX), min(h - 1, stopY)
            # Grab the ROI, convert to RGB ordering, resize and then preprocess
            face = img[beginY:stopY, beginX:stopX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (112, 112))
            face = img_to_array(face)
            face = preprocess_input(face)
            # Expand to fit the image
            face = expand_dims(face, axis = 0)
            # Model prediction order --> incorrect_mask, with_mask, without_mask
            preds = mask_net.predict(face)[0]
            label = np.argmax(preds)
            
            if label == 0:
                cv2.putText(img, f"Incorrect Mask {(preds[0]*100):.2f}%", (beginX-2, beginY-8), config.FONT, 0.5, (15, 50, 100), 2)
                cv2.rectangle(img, (beginX, beginY), (stopX, stopY), (15, 50, 100), 2)
            if label == 1:
                cv2.putText(img, f"Mask {(preds[1]*100):.2f}%", (beginX-2, beginY-8), config.FONT, 0.5, (0, 255, 0), 2)
                cv2.rectangle(img, (beginX, beginY), (stopX, stopY), (0, 255, 0), 2)
            else:
                cv2.putText(img, f"No Mask {(preds[2]*100):.2f}%", (beginX-2, beginY-8), config.FONT, 0.5, (0, 0, 255), 2)
                cv2.rectangle(img, (beginX, beginY), (stopX, stopY), (0, 0, 255), 2)


    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    