import os
import warnings

import cv2
import dlib

warnings.filterwarnings('ignore')

# global value
path = 'Dataset'
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


def add_faces(n_faces, dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
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
            if os.path.isdir(f"{dirname}/{name}"):
                for file in os.listdir(f"{dirname}/{name}"):
                    os.remove(f"{dirname}/{name}/{file}")
                os.removedirs(f"{dirname}/{name}")
            return 'No one'
        if key % 256 == 32:  # space pressed
            if len(bounding_box) == 1:
                left, top, right, bottom = bb_coord(bounding_box[0])
                while flag:
                    print("Enter your name: ")
                    name = input()
                    if os.path.isdir(f"{dirname}/{name}"):
                        print("Already exist")
                    else:
                        os.mkdir(f"{dirname}/{name}")
                        flag = False
                crop = gray[top:bottom, left:right]
                resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"{dirname}/{name}/{count}.jpg", resized)
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
    print('aaaaaaa')
    dirname = "Dataset"
    name = add_faces(10, dirname)
    print(f"{name} has added faces to the {dirname}")


if __name__ == '__main__':
    main()
