import cv2
import numpy as np
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recogniser/trainingData.yml')
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img = cam.read();
    if not img is None:
        if not ret: continue
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray) #return coordinate of faces

        for (x,y,w,h) in faces: # draw a rectangle in each faces
            id,conf = rec.predict(gray[x:x+w,y:y+h])
            if (id == 1):
                id="Aniket"
            elif (id == 2):
                id = "Sanju"
            cv2.putText(img,str(id),(x,y+h),font,1,255,2,1)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Face",img)
        if (cv2.waitKey(1) == ord('q')): #break operation on q char
            break;
cam.release()
cv2.destroyAllWindows()
