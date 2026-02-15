import cv2
import pandas as pd
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Trainer/trainer.yml')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

cam = cv2.VideoCapture(0)

def mark_attendance(id):
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date = now.strftime("%Y-%m-%d")

    with open("attendance.csv", "a") as f:
        f.write(f"{id},{date},{time}\n")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if confidence < 100:
            mark_attendance(id)
            label = f"Employee {id}"
        else:
            label = "Unknown"

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(255,255,255),2)

    cv2.imshow("Recognition", img)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
