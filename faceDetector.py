import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # ubah lebar cam
cam.set(4, 480)  # ubah tinggi cam

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eyesDetector = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(
        abuAbu, 1.3, 5)  # frame, scalefactor, minNeighbors
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
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
cam.release()
cv2.destroyAllWindows()
