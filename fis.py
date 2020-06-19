import argparse
import os
import pickle
import warnings

import cv2
import dlib
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

parser = argparse.ArgumentParser(description='FIS')
parser.add_argument('--type', type=int, default=0, help='0: FIS, 1: add a new guy and retrain')
parser.add_argument('--name', type=str, default='Anonymous', help='name of new guy')
args = parser.parse_args()



warnings.filterwarnings('ignore')

# global value
ppc = (8, 8)  # pixels per cell
cpb = (2, 2)  # cell per block
width = 800
height = 800


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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    count = 0

    name = args.name
    if len(name) == 0:
        name = 'Anonymous'
    else:
        name = name.replace('_', ' ')
    if os.path.isdir(f"{dirname}/{name}"):
        tmp = 1
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
        cv2.imshow(dirname, flip_frame)

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
            # while len(bounding_box) > 0 and count < n_faces:
            #     if len(bounding_box) == 1:
            #         left, top, right, bottom = bb_coord(bounding_box[0])
            #         crop = gray[top:bottom, left:right]
            #         resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
            #         cv2.imwrite(f"{dirname}/{name}/{count}.jpg", resized)
            #         count += 1
            #     else:
            #         print("Only 1 face per time")
    cap.release()
    cv2.destroyAllWindows()
    return name


def FIS(clf):
    print('Camera warming up...')
    hog = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)  # ith webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    status = True
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_box = hog(gray, 0)
        for box in bounding_box:
            status = False
            left, top, right, bottom = bb_coord(box)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
            crop = gray[top:bottom, left:right]
            try:
                resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
                temp = [resized]
                fd = feature_descriptor(temp)
                label = str(clf.predict(fd))
                label = label[2:-2]
                if len(bounding_box) > 0:
                    if not status:
                        flip_frame = cv2.flip(frame, 1)
                        x = width - int((right + left) / 2) - 250
                        y = bottom
                        cv2.putText(flip_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
                        cv2.imshow("Facial Identification System", flip_frame)
            except:
                status = True
                break
        if len(bounding_box) == 0:
            status = True
        if status:
            flip_frame = cv2.flip(frame, 1)
            cv2.imshow("Facial Identification System", flip_frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:
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
    clf = svm.SVC(C=5, gamma=0.001)
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
        if args.name == 'evaluation':
            dirname = 'Test'
            data, labels = load_dataset(dirname)
            data_fd = feature_descriptor(data)
            predict = []
            for i in data_fd:
                temp = str(clf.predict(i[np.newaxis]))
                temp = temp[2:-2]
                predict.append(temp)
            print(f"Accuracy:{accuracy_score(labels,predict)}")
        else:
            FIS(clf)
    elif args.type == 1:  # add a new guy and retrain
        dirname = 'Dataset'
        who = add_faces(10, dirname)
        print(f"{who} has added faces to the Dataset")
        # Train
        if who != 'No one':
            data, labels = load_dataset(dirname)
            data_fd = feature_descriptor(data)
            clf = train(data_fd, np.array(labels))
            save_model(clf, 'FIS')
    elif args.type == 2:
        dirname = 'Test'
        add_faces(10, dirname)
    else:
        print(f"Invalid parameter for type:{args.type} \nPlease search for help")


if __name__ == '__main__':
    main()
