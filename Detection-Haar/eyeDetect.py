import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cv2.namedWindow("Frame")
cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    neighbours = cv2.getTrackbarPos("Neighbours", "Frame")
    
    faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, neighbours)
    
    for rect in faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for rect_eye in eyes:
            (ex,ey,ew,eh) = rect_eye
            frame = cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (150, 255, 0), 2)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
