import face_recognition
import cv2
import numpy as np
import os

#Import the images from the directory
path = 'D:/Jaunel/Learning/DL/computer vision/face Recognition/ImagesAttendence'
images = []                       #for storing all the images in the list
classNames = []                   # for storing name of all the stored images
myList = os.listdir(path)         # list down all files in the folder
print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')    #fetching images one by one
    images.append(curImg)                   #appending the image list
    classNames.append(cls.split('.')[0])    #storing first name in the nameList
print(classNames)
print(len(images))

#Encoding all the stored images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]    #encoding images one by one
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

#Initializing webcam for compairing task
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()                                                       #reading frames of images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                     #converting to RGB format
    facesCurFrame = face_recognition.face_locations(img)                            #locating face in the current frame
    encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame)           #encoding faces in the current frame
    '''Now, matching the encodings in the current frame with the known encodings'''
    for encodeFace in encodesCurFrame:
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)         #comparing faceencoding in frame with known encodings
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)         #calculating distance between these encodings
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            for faceLoc in facesCurFrame:
                y1,x2,y2,x1 = faceLoc
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)                     #drawing rectangle foe face
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)         #drawing rectangle for text
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #writinginside the box


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
