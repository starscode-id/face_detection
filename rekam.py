import cv2
import os
wajahDir = 'data'
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # ubah lebar cam
cam.set(4, 480)  # ubah tinggi cam

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eyesDetector = cv2.CascadeClassifier('haarcascade_eye.xml')

faceID = input("masukkan face id yang akan di rekam: ")
print("silahkan fokuskan wajah anda ke arah kamera, proses pengambilan data akan di lakukan")
ambilData = 1
while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)  # flip video image vertically
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(
        abuAbu, 1.3, 5)  # frame, scalefactor, minNeighbors
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        namaFile = 'wajah.'+str(faceID)+'.'+str(ambilData)+'.jpg'
        ambilData += 1
        cv2.imwrite(wajahDir+'/'+namaFile, frame)
        roiAbuAbu = abuAbu[y:y+h, x:x+w]
        roiWarna = frame[y:y+h, x:x+w]
        eyes = eyesDetector.detectMultiScale(roiAbuAbu)
        for (xe, ye, we, he) in eyes:
            cv2.rectangle(roiWarna, (xe, ye), (xe+we, ye+he), (0, 0, 255), 1)
    cv2.imshow('webcam', frame)
    # cv2.imshow('webcam-grey', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif ambilData >= 30:
        break
print("pengambilan data selesai")
cam.release()
cv2.destroyAllWindows()
