import cv2
import numpy as np
import face_recognition
import os 
from datetime import datetime

def get_data(path):
    # path = 'attendance_proj/classinfo'
    images = []
    className = []
    myList = os.listdir(path) 
    for cls in myList:
        curImg = cv2.imread(path + '/' + cls)
        images.append(curImg)
        className.append(os.path.splitext(cls)[0]) 
    return images, className

# only for the class

def findEncoding(images):
    encodelist =[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markattendance(name):
    with open ('attendance_proj\Attendance.csv', 'r+') as f: #r+ mode allows reading and writing
        mydata = f.readlines()
        namelist = []
        for line in mydata:
            entry = line.split(',')
            namelist. append(entry[0])
            if name not in namelist:
                now = datetime.now()
                dtstring = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name}, {dtstring}')


def verifynew(new_images,list_encode, class_name):
    # new_images = cv2.imread(new_images)
    img = cv2.resize(new_images,(0,0),None,0.25,0.25) #if set the proportion to (0,0), then fx, fy used instead
    img = cv2.cvtColor(new_images,cv2.COLOR_BGR2RGB)
    face_location = face_recognition.face_locations(img)
    encode_face = face_recognition.face_encodings(img,face_location)

    for encodef, floc in zip (encode_face, face_location):
        matches = face_recognition.compare_faces(list_encode, encodef)
        faceDis = face_recognition.face_distance(list_encode,encodef)
        matching_number = np.argmin (faceDis) #return index of the minimun value

    # if matches[matching_number]: #return true/false
    #     name = class_name[matching_number].upper()
    #     y1,x2,y2,x1 = floc
    #     y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 #scaled down then scaled up 
    #     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    #     cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
    #     cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    #     markattendance (name)
        if faceDis[matching_number]< 0.50:
            name = class_name[matching_number].upper()
            markattendance(name)
        else: 
            name = 'Unknown'
        y1,x2,y2,x1 = floc
        y1, x2, y2, x1 = y1,x2,y2,x1
        cv2.rectangle(new_images,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(new_images,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(new_images,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.imshow('Webcam',new_images)
        cv2.waitKey(1)






