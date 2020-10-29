import os
import imutils
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class FaceEmbedder:
    def __init__(self):
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join(["models", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["models", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.face_detection_model = cv2.dnn.readNet(prototxtPath, weightsPath)
        print("[INFO] loading face embedding model...")
        face_embedding_model_path = os.path.sep.join(["models", "facenet_keras.h5"])
        self.face_embedding_model = load_model(face_embedding_model_path)

    def scanImage(self, image):
        """
            Image must be given as a RGB numpy array. Returns a 128 vector representing the face
        """
        locs = self.__detect_face__(image, self.face_detection_model, 0.5)
        is_someone = bool(len(locs) > 0)
        if is_someone:
            #Extract face from image
            widths = [loc[2] - loc[0] for loc in locs]
            index = widths.index(max(widths))
            (startX, startY, endX, endY) = locs[index]
            face = image[startY:endY, startX:endX]
            #Process face
            face = cv2.resize(face, (160, 160))
            face = face.reshape(1,*face.shape)
            pred = self.face_embedding_model.predict(face)
            return pred[0]
        else:
            return []

    def __detect_face__(self,frame, faceNet, confidence_accepted=0.5):
        (h, w) = frame.shape[:2]
        frame = imutils.resize(frame, width=400)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        locs = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_accepted:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                locs.append((startX, startY, endX, endY))
        return locs
