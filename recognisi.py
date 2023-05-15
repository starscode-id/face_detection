import cv2
import os
import numpy as np
wajahDir = 'data'
trainDir = 'train'
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
names = ["tidak diketahui", "idris", "galih", "raja"]
faceRecognizer.read(trainDir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(
        abuAbu, 1.2, 5, minSize=(int(minWidth), int(minHeight)),)  # frame, scalefactor, minNeighbors
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = faceRecognizer.predict(abuAbu[y:y+h, x:x+w])
        if confidence <= 50:
            nameID = names[id]
            confidencetText = "{0}".format(round(100-confidence))
        else:
            nameID = names[0]
            confidencetText = "{0}".format(round(100-confidence))

        cv2.putText(frame, str(nameID), (x+5, y-5),
                    font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidencetText),
                    (x+5, y-5), font, 1, (255, 255, 0), 1)
        # if (id == 1):
        #     id = 'idris'
        # elif (id == 2):
        #     id = 'galih'
        # elif (id == 3):
        #     id = 'raja'
        # else:
        #     id = 'tidak diketahui'
        # cv2.putText(frame, str(id), (x+5, y-5),
        #             font, 1, (255, 255, 255), 2)
    cv2.imshow('recognisi wajah', frame)
    # cv2.imshow('webcam-grey', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

print("exit")
cam.release()
cv2.destroyAllWindows()
