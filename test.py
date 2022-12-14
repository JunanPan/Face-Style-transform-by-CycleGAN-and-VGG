import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess


parser = argparse.ArgumentParser()
#拿到传入路径，保存路径以及模型路径，就可以进行预测了
parser.add_argument('--photo_path', type=str, help='input photo path')
parser.add_argument('--save_path', type=str, help='cartoon save path')
parser.add_argument('--model', type=str, help='cartoon save path')
args = parser.parse_args()
#如果保存路径不存在 就创建一个文件夹
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = ResnetGenerator(ngf=28, img_size=256, light=True).to(self.device)
        if args.model=='sketch':
            params = torch.load('D:\experiment/train-size256-ch28-True-lr0.0001-adv1-cyc50-id1-identity10-cam1000\photo2sketch_params_latest.pt', map_location=self.device)
            self.net.load_state_dict(params['genA2B'])
        elif args.model=='cartoon':
            params = torch.load('D:\experiment/train-size256-ch28-True-lr0.0001-adv1-cyc50-id1-identity10-cam1000\photo2cartoon_params_latest.pt', map_location=self.device)
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
if __name__ == '__main__':
    #cv2.imread()接口读图像,读进来直接是BGR
    img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
    c2p = Photo2Cartoon()
    cartoon = c2p.inference(img)
    if cartoon is not None:
        cv2.imwrite(args.save_path, cartoon)
