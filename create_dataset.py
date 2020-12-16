# -*-coding:utf-8-*-
import argparse
import cv2
import numpy as np
import os
import imageio
import face_recognition
from math import floor, ceil
from PIL import Image
from scipy.ndimage.interpolation import zoom, rotate


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='demo', help='input video path')
parser.add_argument('--output_path', default='demo', help='extracted frame save path')
parser.add_argument('--img_size', default=256, help='resize image size')
opt = parser.parse_args()


class FaceFinder:
    def __init__(self, video_path, load_first_face=True):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.length = int(self.video_capture.get(7))  # cv2.CAP_PROP_FRAME_COUNT
        self.fps = int(ceil(self.video_capture.get(5)))  # cv2.CAP_PROP_FPS
        self.time = int(ceil(self.length / self.fps))
        self.last_location = (0, 200, 200, 0)
        self.faces = {}
        self.coordinates = {}
        self.frame_shape = [self.video_capture.get(4), self.video_capture.get(3)]
        if load_first_face:
            _, frame = self.video_capture.read()
            face_positions = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
            if len(face_positions) > 0:
                self.last_location = self.pop_largest_location(face_positions)
                print("last location is : ", self.last_location)

    def get(self, key):
        self.video_capture.set(1, key)
        success, img = self.video_capture.read()
        return success, img

    def expand_location_zone(self, loc, margin=0.2):
        # Adds a margin around a frame slice 扩展面部区域
        offset = round(margin * (loc[2] - loc[0]))  # face width
        y0 = max(loc[0] - offset, 0)
        x1 = min(loc[1] + offset, self.frame_shape[1])
        y1 = min(loc[2] + offset, self.frame_shape[0])
        x0 = max(loc[3] - offset, 0)
        return (y0, x1, y1, x0)

    @staticmethod
    def upsample_location(reduced_location, upsampled_origin, factor):
        # Adapt a location to an upsampled image slice 调整位置以适应上采样的图像切片
        y0, x1, y1, x0 = reduced_location
        Y0 = round(upsampled_origin[0] + y0 * factor)
        X1 = round(upsampled_origin[1] + x1 * factor)
        Y1 = round(upsampled_origin[0] + y1 * factor)
        X0 = round(upsampled_origin[1] + x0 * factor)
        return (Y0, X1, Y1, X0)

    @staticmethod
    def pop_largest_location(location_list):
        max_location = location_list[0]
        max_size = 0
        if len(location_list) > 1:
            for location in location_list:
                size = location[2] - location[0]
                if size > max_size:
                    max_size = size
                    max_location = location
        return max_location

    @staticmethod
    def L2(A, B):
        return np.sqrt(np.sum(np.square(A - B)))

    # 返回脸部中心点、长度、旋转角度
    def find_coordinates(self, landmark, K = 2.2):
        E1 = np.mean(landmark['left_eye'], axis=0)
        E2 = np.mean(landmark['right_eye'], axis=0)
        E = (E1 + E2) / 2
        N = np.mean(landmark['nose_tip'], axis=0) / 2 + np.mean(landmark['nose_bridge'], axis=0) / 2
        B1 = np.mean(landmark['top_lip'], axis=0)
        B2 = np.mean(landmark['bottom_lip'], axis=0)
        B = (B1 + B2) / 2

        C = N
        l1 = self.L2(E1, E2)
        l2 = self.L2(B, E)
        l = max(l1, l2) * K
        if B[1] == E[1]:
            if B[0] > E[0]:
                rot = 90
            else:
                rot = -90
        else:
            rot = np.arctan((B[0] - E[0]) / (B[1] - E[1])) / np.pi * 180
        return ((floor(C[1]), floor(C[0])), floor(l), rot)

    def find_faces(self, resize=0.5, no_face_acceleration_threshold=3, cut_left=0, cut_right=-1, read_stop=0):
        i = 0
        not_found = 0
        no_face = 0
        no_face_acc = 0
        if read_stop != 0:
            finder_frame_set = range(0, min(self.length, read_stop), 1)  # 取前10帧
        else:
            finder_frame_set = range(0, self.length, self.fps)

        for i in finder_frame_set:
            success, frame = self.get(i)
            if success:
                if cut_left != 0 or cut_right != -1:
                    frame[:, :cut_left] = 0
                    frame[:, cut_right:] = 0
                potential_location = self.expand_location_zone(self.last_location)
                try:
                    potential_face_patch = frame[potential_location[0]:potential_location[2],
                                           potential_location[3]:potential_location[1]]
                except TypeError:
                    continue
                potential_face_patch_origin = (potential_location[0], potential_location[3])  # [x1, y2]
                '''
                缩放会导致脸部照片变的更加模糊，但是这个缩放的目的是为了在face_recognition.face_locations时能够
                更快的找到脸部，糊就说明图片像素变少了，找的速度就快，但是这也有个潜在的问题就是，如果在视频已经经受
                很严重的压缩情况下，可能根本无法从这一步提取到面部，所以，如果在H.264 40压缩质量时，这个操作得去掉
                '''
                reduced_potential_face_patch = zoom(potential_face_patch, (resize, resize, 1))  # cut in half
                reduced_face_locations = face_recognition.face_locations(reduced_potential_face_patch, model='cnn')  # [x1, y1, x2, y2]

                if len(reduced_face_locations) > 0:
                    no_face_acc = 0
                    reduced_face_location = self.pop_largest_location(reduced_face_locations)
                    face_location = self.upsample_location(reduced_face_location, potential_face_patch_origin, 1/resize)
                    print('face extract success in %d frame, location is: ' % i, face_location)
                    self.faces[i] = face_location
                    self.last_location = face_location
                    print("last location is : ", self.last_location)
                    landmarks = face_recognition.face_landmarks(frame, [face_location])
                    if len(landmarks) > 0:
                        # assume that there is one and only one landmark group
                        self.coordinates[i] = self.find_coordinates(landmarks[0])
                else:
                    not_found += 1
                    if no_face_acc < no_face_acceleration_threshold:
                        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
                    else:
                        reduced_frame = zoom(frame, (resize, resize, 1))
                        face_locations = face_recognition.face_locations(reduced_frame)
                    if len(face_locations) > 0:
                        print('Face extraction warning: ', i, ' frame, found face in full frame', face_locations)
                        no_face_acc = 0
                        face_location = self.pop_largest_location(face_locations)
                        if no_face_acc > no_face_acceleration_threshold:
                            face_location = self.upsample_location(face_location, (0, 0), 1/resize)

                        self.faces[i] = face_location
                        self.last_location = face_location
                        print("last location is : ", self.last_location)
                        # extract face rotation, length and center from landmarks
                        landmarks = face_recognition.face_landmarks(frame, [face_location])
                        if len(landmarks) > 0:
                            self.coordinates[i] = self.find_coordinates(landmarks[0])
                    else:
                        print('Face extraction warning: ', i, 'no face')
                        no_face_acc += 1
                        no_face += 1

    @staticmethod
    def get_image_slice(img, y0, y1, x0, x1):
        # Get values outside the domain of an image 获取图像外部的值
        m, n = img.shape[:2]  # m-height n-width
        padding = max(-y0, y1-m, -x0, x1-n, 0)
        padded_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
        return padded_img[(padding + y0):(padding + y1),
                        (padding + x0):(padding + x1)]

    def get_aligned_face(self, i, l_factor = 1.3):
        _, frame = self.get(i)
        if i in self.coordinates:
            c, l, r = self.coordinates[i]  # stores the face (locations center, rotation, length)
            l = int(l) * l_factor # fine-tuning the face zoom we really want
            dl_ = floor(np.sqrt(2) * l / 2)  # largest zone even when rotated
            patch = self.get_image_slice(frame,
                                    floor(c[0] - dl_),
                                    floor(c[0] + dl_),
                                    floor(c[1] - dl_),
                                    floor(c[1] + dl_))
            rotated_patch = rotate(patch, -r, reshape=False)
            # note : dl_ is the center of the patch of length 2dl_
            return self.get_image_slice(rotated_patch,
                                    floor(dl_-l//2),
                                    floor(dl_+l//2),
                                    floor(dl_-l//2),
                                    floor(dl_+l//2))
        return frame


class FaceWriter:
    def __init__(self, face_finder, img_size=256):
        self.finder = face_finder
        self.img_size = img_size
        self.length = int(face_finder.length)
        self.head = 0

    # enlarge the face image
    def resize_patch(self, patch):
        m, n = patch.shape[:2]
        return zoom(patch, (self.img_size/m, self.img_size/n, 1))

    def img_writer(self, video_name, save_path):
        while self.head < self.length:
            if self.head in self.finder.coordinates:
                patch = self.finder.get_aligned_face(self.head)
                img = self.resize_patch(patch)
                # img = Image.fromarray(img)
                # img.save('{}/{}{}.jpg'.format(save_path, video_name, self.head))
                cv2.imwrite('{}/{}_{}.jpg'.format(save_path, video_name, self.head), img)
            self.head += self.finder.fps
        self.finder.video_capture.release()


def produce_face_img(input_path, output_path, img_size):
    video_type = ['.mp4', '.avi', '.mov']
    video_files = []
    for f in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, f)) and (f[-4:] in video_type):
            video_files.append(os.path.join(input_path, f))

    for video in video_files:
        print("extract face from %s" % video)
        video_name = os.path.splitext(os.path.basename(video))[0]
        face_finder = FaceFinder(video)
        face_finder.find_faces()
        print('The video {} processing is completed, and a total of {} facial images are extracted'
              .format(video_name, len(face_finder.faces)))
        face_writer = FaceWriter(face_finder, img_size)
        face_writer.img_writer(video_name, output_path)


if __name__ == '__main__':
    print(opt)
    produce_face_img("E:\\DeepLearning\\dataset\\FaceForensics++\\original_sequences\\youtube\\c23\\videos",
                     "datasets/deepfakes/train/original",
                     opt.img_size)
