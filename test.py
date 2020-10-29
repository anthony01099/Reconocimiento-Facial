import cv2, time
import numpy as np
from face_embedding import FaceEmbedder

#Flags
show_vector = False
time_between_photos = 5

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
#Start face embedding
face_embedder = FaceEmbedder()
print('Done')

previous_vector = []
while True:
    print('Taking picture.')
    success, frame = vs.read()
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_vector = face_embedder.scanImage(frame)
        if len(face_vector) > 0:
            if show_vector:
                print(face_vector)
            if len(previous_vector):
                dist = np.linalg.norm(face_vector - previous_vector)
                print("Difference score respect to previous face: {:.2f} ".format(dist))
            previous_vector = face_vector

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(time_between_photos)
