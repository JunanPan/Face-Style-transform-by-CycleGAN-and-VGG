import cv2
import math
import numpy as np
import face_alignment#可以进行人脸关键点检测


class FaceDetect:
    def __init__(self, device, detector):
        # 人脸关键点检测对象
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)

    def align(self, image):
        landmarks = self.__get_max_face_landmarks(image)

        if landmarks is None:
            return None

        else:
            return self.__rotate(image, landmarks)

    def __get_max_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)
        if preds is None:
            return None

        elif len(preds) == 1:#如果只有一张脸 ，直接返回这张
            return preds[0]

        else:
            # 如果有多张，找到最大的那张脸
            areas = []
            for pred in preds:#对每套关键点，计算面积，并添加进列表
                landmarks_top = np.min(pred[:, 1])#顶部取min
                landmarks_bottom = np.max(pred[:, 1])#底部取max
                landmarks_left = np.min(pred[:, 0])#左边取min
                landmarks_right = np.max(pred[:, 0])#右边取max
                #计算面积
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)#找出最大的area对应的下标
            return preds[max_face_index]

    @staticmethod #旋转图像，微调正脸
    def __rotate(image, landmarks):
        # 得到左眼和右眼的坐标
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        #找到翻转角度 用arctan函数
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        # 旋转后的图像大小
        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # 仿射变换矩阵
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        return image_rotate, landmarks_rotate


