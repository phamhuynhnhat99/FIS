import os

import cv2
import dlib


def a():
    img = cv2.imread('face.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = dlib.get_frontal_face_detector()

    bounding_box = hog(gray, 0)
    print(len(bounding_box))
    if len(bounding_box) == 1:
        face = bounding_box[0]
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        w = face.right() - face.left()
        h = face.bottom() - face.top()
        crop = gray[top:bottom, left:right]
        print(left, right, top, bottom)
        for face in bounding_box:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('sas', gray)
    #cv2.putText(crop, "daaaaaaaaaaaaa", (int((bounding_box[0].right() - bounding_box[0].right()))/2, bounding_box[0].top() -1 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 2,)
    cv2.imshow('we', crop)
    cv2.waitKey()

    while (True):
        print("Enter your name: ")
        name = input()
        if os.path.isdir(f"Dataset/{name}"):
            print("Already exist")
        else:
            os.mkdir(f"Dataset/{name}")
            break


labels = []
for name in os.listdir('Dataset'):
    labels.append(name)
data = [[]]
a()
