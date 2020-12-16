# -*-coding:utf-8-*-
import cv2
import os
import dlib
import numpy as np
import time
import scipy
from scipy import constants as C
from tqdm import tqdm
from tqdm._tqdm import trange
from math import ceil
import torch
import torchvision.transforms as transforms
from torch.nn.modules import Conv2d


def show_im(input):
    img = cv2.imread(input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask_cropped = np.stack((binary,binary, binary), axis=2)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    locations = np.array([], dtype=np.int).reshape(0, 5)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    print(contours)
    cv2.imshow("img", img)


def latent_vector():
    img = cv2.imread("./datasets/test/F_FAP1_00334-1.png")
    img = cv2.resize(img, (256, 256))

    transf = transforms.ToTensor()
    img = transf(img)

    latent = torch.reshape(img, (-1, 2, 64, 16, 16))

    zero_abs = torch.abs(latent[:,0]).view(latent.shape[0], -1)
    zero = zero_abs.mean(dim=1)

    one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
    one = one_abs.mean(dim=1)

    labels = np.array([1])
    y = torch.eye(2)
    y = y.index_select(dim=0, index=torch.tensor(0))
    latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)
    print(latent.shape)


def select_index():
    a = torch.linspace(1, 12, steps=12).view(3, 4)
    print(a)
    # b = torch.index_select(a, 1, torch.tensor([0, 1, 2, 2]))
    # print(b.shape)
    # print(b)
    # print("\n")
    # print(b[:, :, None, None, None])
    y = torch.eye(2)

    y = y.index_select(dim=0, index=torch.tensor([0, 1, 1, 0]))
    print(y)
    print(y[:, :, None, None])
    b = a * y[:, :, None, None]

    print(b.shape)


def gaussian_blur():
    img = cv2.imread("test_img/fake.jpg")
    # blur = cv2.GaussianBlur(img, (7, 7), 1)
    blur = cv2.bilateralFilter(img,13,46,8)  # 双边滤波，能够保持边界特性
    both = np.vstack((img, blur))
    cv2.imshow("", both)
    cv2.waitKey()


def to_bw(mask, thresh_binary=10, thresh_otsu=255):
    im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # cv2.imread()读取的是BGR格式，cv2.cvtColor用于颜色空间转换
    # cv2.threshold，第一个参数是二值图，第二个是阈值(起始值)，第三个是最大值，第四个是划分的算法，这两个算法都用能够返回较精细的分割
    # 返回的是阈值(就是传进去的阈值参数) 和阈值化后的图片
    (thresh, im_bw) = cv2.threshold(im_gray, thresh_binary, thresh_otsu, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return im_bw


def _roc_curve():
    import numpy as np
    from sklearn.metrics import roc_curve
    from matplotlib import pyplot as plt

    y = np.array([1, 1, 2, 2])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
    plt.plot(fpr, tpr, linewidth=2, label="ROC")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.ylim(0.1, 1)
    plt.xlim(0.1, 1)
    plt.legend(loc=4)
    plt.show()


def load_facedetector():
    face_detector = dlib.get_frontal_face_detector()
    sp68 = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    return face_detector, sp68

# print(np.argmax(model.predict(imgs), axis=1))


def read_video():
    vidcap_real = cv2.VideoCapture("datasets/deepfakes/train/altered/033_097.mp4")
    frame_count = vidcap_real.get(7)
    fps = vidcap_real.get(5)
    print(frame_count, fps)
    print(ceil(frame_count / fps))


if __name__ == '__main__':
    pass

