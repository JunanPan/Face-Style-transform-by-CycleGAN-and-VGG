import os
import cv2
import torch
import sys, os
sys.path.append("..")
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess
import base64
class Photo2Cartoon:
    def __init__(self,mode):
        self.pre = Preprocess()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.mode = mode
        self.net = ResnetGenerator(ngf=28, img_size=256, light=True).to(self.device)
        if self.mode==1:
            params = torch.load('/srv/test/experiment/train-size256-ch28-True-lr0.0001-adv1-cyc50-id1-identity10-cam1000/photo2cartoon_params_latest.pt', map_location=self.device)
            self.net.load_state_dict(params['genA2B'])
        elif self.mode==2:
            params = torch.load('/srv/test/experiment/train-size256-ch28-True-lr0.0001-adv1-cyc50-id1-identity10-cam1000/photo2cartoon1_params_latest.pt', map_location=self.device)
            self.net.load_state_dict(params['genA2B'])
        elif self.mode==3:
            params = torch.load('/srv/test/experiment/train-size256-ch28-True-lr0.0001-adv1-cyc50-id1-identity10-cam1000/photo2sketch_params_latest.pt', map_location=self.device)
            self.net.load_state_dict(params['genA2B'])
    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('can not detect face!!!')
            return None
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1
        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)
        # 进行预测 关闭梯度
        with torch.no_grad():
            cartoon = self.net(face)[0][0]
        # 进一步处理
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))#改变轴的顺序 交换通道
        #进行反归一化等处理
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        return cartoon
#进行图片预测
def predict(input_base64_byte):
    image_data = base64.b64decode(input_base64_byte)
    # 转换为np数组
    image_array = np.fromstring(image_data, np.uint8)
    # 转换成opencv可用格式
    image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)
    #image可以用cv2.imshow()直接显示用于测试或者进行后续处理
    #cv2.imread()接口读图像,读进来直接是BGR
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #现在img就是可以直接用的用于传入模型的input

    prefix = 'data:image/png;base64,'

    c2p = Photo2Cartoon(mode=1)
    cartoon = c2p.inference(img)
    image = cv2.imencode('.jpg',cartoon)[1]
    result_base64_str1 = prefix + str(base64.b64encode(image))[2:-1]

    c2p = Photo2Cartoon(mode=2)
    cartoon = c2p.inference(img)
    image = cv2.imencode('.jpg',cartoon)[1]
    result_base64_str2 = prefix + str(base64.b64encode(image))[2:-1]
    #
    c2p = Photo2Cartoon(mode=3)
    cartoon = c2p.inference(img)
    image = cv2.imencode('.jpg',cartoon)[1]
    result_base64_str3 = prefix + str(base64.b64encode(image))[2:-1]

    return result_base64_str1,result_base64_str2,result_base64_str3
