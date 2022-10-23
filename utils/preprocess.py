from .face_detect import FaceDetect
from .face_seg import FaceSeg
import numpy as np


class Preprocess:
    def __init__(self, device='cpu', detector='dlib'):#dlib是FaceDetect的一个参数选项
        self.detect = FaceDetect(device, detector) 
        self.segment = FaceSeg()

    def process(self, image):
        face_info = self.detect.align(image)#送入facedetect
        if face_info is None:
            return None
        image_align, landmarks_align = face_info#拿到图像和关键点信息，从而方便旋转

        face = self.__crop(image_align, landmarks_align)#进行旋转
        mask = self.segment.get_mask(face)#截取头像区域
        return np.dstack((face, mask))#融合二者结果

    @staticmethod
    def __crop(image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        #稍微扩大一点这个区域
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        #修一下比例
        #如果上下太长了，left减一点，right加一点
        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        #如果左右太长了，top提一点，bottom降一点
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)
        
        #按照上面订好的大小 生成一个框框 初始化都为255 即都是白色
        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        #left_white就代表背景为白色的，左边界处，image_crop用white，image不带white
        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)
        # 给生成好的框框赋值
        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
        return image_crop
