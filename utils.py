import numpy as np
import csv
import cv2


def get_batch(butch_size, train=True):
    """
    gets a butch of training images
    :param butch_size: butch size
    :type butch_size: int
    :param train:if true get train data if false get test data
    :type train: bool
    :return: images and labels
    :rtype: (list, list)
    """
    folder = 'train' if train else 'test'
    images = []
    labels = []
    with open(f'data/{folder}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    idx = np.random.randint(0, len(reader), butch_size)
    for i in idx:
        image = cv2.imread(r"data/" + reader[i][0] + '.jpg')
        label = np.zeros((7, 7, 30))
        images.append(image)
        with open(r"data/" + reader[i][1] + '.txt') as f:
            for line in f.readlines():
                s_w = int(line[1] // len(image) / 7)
                s_h = int(line[2] // len(image[0]) / 7)
                label[s_w, s_h, line[0]] = 1
                if any(label[s_w, s_h, 10:31]):
                    if line[3] / line[4] > label[s_w, s_h, 2] / label[s_w, s_h, 3]:
                        x = label[s_w, s_h, 0:4]
                        label[s_w, s_h, 0:4] = np.array(line[1:5])
                        label[s_w, s_h, 4:8] = x
                    else:
                        label[s_w, s_h, 4:8] = np.array(line[1:5])
                else:
                    label[s_w, s_h, 0:4] = np.array(line[1:5])

        labels.append(label)
    return images, labels
