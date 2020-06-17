import os
import warnings
import cv2
import dlib

from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score


warnings.filterwarnings('ignore')


# global value
path = 'Dataset/'
ppc = 8  # pixels per cell
cpb = 2  # cell per block
hog_images = []
hog_features = []
labels = []
data = []


def bb_coord(face):
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    return left, top, right, bottom


def add_faces(n_faces):
    if not os.path.isdir('Dataset'):
        os.mkdir('Dataset')
    hog = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)  # ith webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    count = 0
    flag = True
    name = None
    while count < n_faces:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_box = hog(gray, 0)
        if len(bounding_box) == 1:
            left, top, right, bottom = bb_coord(bounding_box[0])
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 3)

        flip_frame = cv2.flip(frame, 1)
        cv2.putText(flip_frame, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255))
        cv2.imshow("Webcam", flip_frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:  # escape pressed
            if os.path.isdir(f"Dataset/{name}"):
                for file in os.listdir(f"Dataset/{name}"):
                    os.remove(f"Dataset/{name}/{file}")
                os.removedirs(f"Dataset/{name}")
            return 'No one'
        if key % 256 == 32:  # space pressed
            if len(bounding_box) == 1:
                left, top, right, bottom = bb_coord(bounding_box[0])
                while flag:
                    print("Enter your name: ")
                    name = input()
                    if os.path.isdir(f"Dataset/{name}"):
                        print("Already exist")
                    else:
                        os.mkdir(f"Dataset/{name}")
                        flag = False
                crop = gray[top:bottom, left:right]
                resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"Dataset/{name}/{count}.jpg", resized)
                count += 1
            else:
                print("Only 1 face per time")
    cap.release()
    cv2.destroyAllWindows()
    return name


def load_dataset(path):
    labels = []
    data = []
    for name in os.listdir(path):
        for file in os.listdir(f"{path}/{name}"):
            img = cv2.imread(f"{path}/{name}/{file}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            labels.append(name)
            data.append(gray)
    return data, labels


def train():
    data, labels = load_dataset(path)
    pass


def main():
    name = add_faces(10)
    print(f"{name} has added faces to the Dataset")



if __name__ == '__main__':
    main()
