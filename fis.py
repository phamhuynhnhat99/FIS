import os
import pickle
import warnings

import cv2
import dlib
import numpy as np
from skimage.feature import hog
from sklearn import svm

import argparse

parser = argparse.ArgumentParser(description='FIS')
parser.add_argument('--type', type=int, default=0, help='0: FIS, 1: add a new guy and retrain')
parser.add_argument('--name', type=str, default='Anonymous', help='name of new guy')
args = parser.parse_args()

# from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# global value
ppc = (8, 8)  # pixels per cell
cpb = (2, 2)  # cell per block


def bb_coord(face):
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    return left, top, right, bottom


def add_faces(n_faces, dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    hog = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)  # ith webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    count = 0

    name = args.name
    if len(name) == 0:
        name = 'Anonymous'
    tmp = 0
    while os.path.isdir(f"{dirname}/{name}" + '_' + str(tmp)):
        tmp += 1
    name = name + '_' + str(tmp)
    os.mkdir(f"{dirname}/{name}")

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
            if os.path.isdir(f"{dirname}/{name}"):
                for file in os.listdir(f"{dirname}/{name}"):
                    os.remove(f"{dirname}/{name}/{file}")
                os.removedirs(f"{dirname}/{name}")
            return 'No one'
        if key % 256 == 32:  # space pressed
            if len(bounding_box) == 1:
                left, top, right, bottom = bb_coord(bounding_box[0])

                crop = gray[top:bottom, left:right]
                resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"{dirname}/{name}/{count}.jpg", resized)
                count += 1
            else:
                print("Only 1 face per time")
    cap.release()
    cv2.destroyAllWindows()
    return name


def FIS(clf):
    hog = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)  # ith webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    result = None

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_box = hog(gray, 0)

        flip_frame = cv2.flip(frame, 1)
        for ind, box in enumerate(bounding_box):
            left, top, right, bottom = bb_coord(box)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

            crop = gray[top:bottom, left:right]
            resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)

            fd_resize = hog(resized, orientations=9, pixels_per_cell=ppc, cells_per_block=cpb, visualize=False,
                            multichannel=False)
            fd_resize = np.array(fd_resize)
            label = clf.predict(fd_resize)
            print(type(label))
            print(label)

            # cv2.putText(flip_frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 255, 255))
        cv2.imshow("Facial Identification System", flip_frame)
        
        key = cv2.waitKey(1)
        if key % 256 == 32:
            break

    cap.release()
    cv2.destroyAllWindows()


def load_model(path):
    if os.path.exists(path):
        model = pickle.load(open(path, 'rb'))
        return model
    print("Model not found")
    return None


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


def feature_descriptor(data):
    features = []
    for img in data:
        fd = hog(img, orientations=9, pixels_per_cell=ppc, cells_per_block=cpb, visualize=False, multichannel=False)
        features.append(fd)
    return np.array(features)


def train(data_fd, labels):
    clf = svm.SVC()
    clf.fit(data_fd, labels)
    return clf


def save_model(model, name):
    if not os.path.isdir("Model"):
        os.mkdir("Model")
    path = f"Model/{name}.sav"
    if os.path.exists(path):
        os.remove(path)
    pickle.dump(model, open(path, 'wb'))
    print(f"Model has added {name}")


def main():
    if args.type == 0:  # test
        clf = load_model('Model/FIS.sav')
        FIS(clf)
    else:  # add a new guy and retrain
        dirname = 'Dataset'
        add_faces(10, dirname)
        # retrain
        data, labels = load_dataset(dirname)
        data_fd = feature_descriptor(data)
        clf = train(data_fd, np.array(labels))
        save_model(clf, 'FIS')


if __name__ == '__main__':
    main()
