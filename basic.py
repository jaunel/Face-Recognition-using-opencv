#pip install cmake
#pip install dlib
#pip install face_recognition

import face_recognition
import cv2
import numpy as np

#loading the  training image
imgElon = face_recognition.load_image_file('D:/Jaunel/Learning/DL/computer vision/face Recognition/ImageBasic/elon1.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)         #converting to RGB format
faceLoc = face_recognition.face_locations(imgElon)[0]        #find the face in image - returns list having coordinates of face
encodeElon = face_recognition.face_encodings(imgElon)[0]     #encoding the image - returns list having encodings of face
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),
              (faceLoc[1],faceLoc[2]),(0,255,0),2)          #drawing rectangle around the face


#loading the  test image
imgTest = face_recognition.load_image_file('D:/Jaunel/Learning/DL/computer vision/face Recognition/ImageBasic/bill1.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
faceTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceTest[3],faceTest[0]),
              (faceTest[1],faceTest[2]),(0,255,0),2)

'''Compairing these faces & finding the distance between them
Backend we will use linear svm to find out whether they match or not'''

result = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)  #finding distance between them
cv2.putText(imgTest,f"{result} {round(faceDis[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)  #writing text on image
print(result,faceDis)
cv2.imshow("Elon train",imgElon)
cv2.imshow("Elon test",imgTest)
cv2.waitKey(0)