import numpy as np
import tensorflow as tf
import os
import cv2
from random import shuffle
import math

#이미지 크기를 60 * 60 으로하고 5종류의 동물을 사용
IMG_HEIGHT = 60 
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5

def load_image(addr):
    img = cv2.imread(addr)
    #리사이즈 보간법 종류: INTER_AREA, INTER_LINEAR
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #데이터 타입: uint8 -> float32
    img = img.astype(np.float32)

    return img

#동물 디렉토리의 파일과 목록을 읽음
IMAGE_DIR_BASE = 'D:/jaemin/Study/deeplearning/image_classification/animals'
image_dir_list = os.listdir(IMAGE_DIR_BASE)

features = list()
labels = list()

#동물별 파일들을 읽음 
for cls_index, dir_name in enumerate(image_dir_list):
    image_file_list = os.listdir(IMAGE_DIR_BASE + os.sep + dir_name)
    for file_name in image_file_list:
        image = load_image(IMAGE_DIR_BASE + os.sep + dir_name + os.sep + file_name)

        #ravel(): n차원 -> 1차원, cls_index -> label
        features.append(image.ravel())
        labels.append(cls_index)

#features, labels 리스트를 셔플후 tuple형태로 -> np array로 반환
shuffle_data = True

if shuffle_data:
    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)

features = np.array(features)
labels = np.array(labels)


#데이터 train set : test set = 8:2 no validation data set
train_features = features[0:int(0.8 * len(features))]
train_labels = labels[0:int(0.8 * len(labels))]

# val_features = features[int(0.6 * len(features)):int(0.8 * len(features))]
# val_labels = labels[int(0.6 * len(features)):int(0.8 * len(features))]

test_features = features[int(0.8 * len(features)):]
test_labels = labels[int(0.8 * len(labels)):]

BATCH_SIZE = 50

def train_data_iterator():
    batch_idx = 0
    while True:

        idxs = np.arange(0, len(train_features))
        np.random.shuffle(idxs)
        shuf_features = train_features[idxs]
        shuf_labels = train_labels[idxs]

        batch_size = BATCH_SIZE

        for batch_idx in range(0, len(train_features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size] / 255.
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch

iter_ = train_data_iterator()

for step in range(100):
    images_batch_val, labels_batch_val = next(iter_)

    print(images_batch_val)
    print(labels_batch_val)

images_batch = tf.placeholder(dtype = tf.float32, shape=[None, IMG_HEIGHT*IMG_WIDTH*NUM_CHANNEL])
labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])

w1 = tf.get_variable("w1", [IMG_HEIGHT*IMG_WIDTH*NUM_CHANNEL, 1024])
b1 = tf.get_variable("b1", [1024])

fc1 = tf.nn.relu(tf.matmul(images_batch, w1) + b1)
