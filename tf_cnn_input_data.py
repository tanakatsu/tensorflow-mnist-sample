# encoding: utf-8
import cv2
import numpy as np
import tf_cnn_model

NUM_CLASSES = tf_cnn_model.NUM_CLASSES
IMAGE_SIZE = tf_cnn_model.IMAGE_SIZE


class DataSets(object):

    def __init__(self, train_data='train.csv', test_data='test.csv', grayScale=False, shuffle=False):
        self.train_data = train_data
        self.test_data = test_data
        self.grayScale = grayScale
        self.shuffle = shuffle

    def read_data_sets(self):
        train_image = []
        train_label = []
        test_image = []
        test_label = []

        with open(self.train_data, "r") as f:
            for line in f:
                line = line.rstrip()
                l = line.split(',')
                l = [w.lstrip() for w in l]
                if self.grayScale:
                    img = cv2.imread(l[0], cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(l[0])
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # width, height
                if not self.grayScale:
                    img = img[:, :, ::-1]  # bgr -> rgb
                train_image.append(img.flatten().astype(np.float32) / 255.0)
                tmp = np.zeros(NUM_CLASSES)
                # ラベルを1-of-k方式で用意する
                tmp[int(l[1])] = 1
                train_label.append(tmp)

        with open(self.test_data, "r") as f:
            for line in f:
                line = line.rstrip()
                l = line.split(',')
                l = [w.lstrip() for w in l]
                if self.grayScale:
                    img = cv2.imread(l[0], cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(l[0])
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # width, height
                if not self.grayScale:
                    img = img[:, :, ::-1]  # bgr -> rgb
                test_image.append(img.flatten().astype(np.float32) / 255.0)
                tmp = np.zeros(NUM_CLASSES)
                tmp[int(l[1])] = 1
                test_label.append(tmp)

        train_image = np.asarray(train_image)
        train_label = np.asarray(train_label)
        test_image = np.asarray(test_image)
        test_label = np.asarray(test_label)

        if self.shuffle:
            perm = np.arange(len(train_image))
            np.random.shuffle(perm)
            train_image = train_image[perm]
            train_label = train_label[perm]

        train = {"images": train_image, "labels": train_label}
        test = {"images": test_image, "labels": test_label}
        return {"train": train, "test": test}

if __name__ == '__main__':
    datasets = DataSets()
    input_data = datasets.read_data_sets()
