import cv2
import os
import numpy as np
from PIL import Image
wajahDir = 'data'
trainDir = 'train'

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    for imagePath in imagePaths:
        PILImg = Image.open(imagePath).convert('L')
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceID)
        return faceSamples, faceID


print("sedang melakukan training data wajah, silahkan tunggu beberapa saat")

faces, IDs = getImageLabel(wajahDir)
faceRecognizer.train(faces, np.array(IDs))
faceRecognizer.write(trainDir+'/training.xml')

print('sebanyak {0} data wajah telah di training ke mesin',
      format(len(np.unique(IDs))))
