
# encoding:utf-8

import requests
import base64
import glob
import os
'''
图像风格转换
'''


def transfer_picture(i):

    # 二进制方式打开图片文件
    f = open('D:\dataset\photo2sketch/testA/'+str(i)+'.png', 'rb')
    img = base64.b64encode(f.read())
    # params = {"image":img,"option":"mononoke"}#修改选项
    params = {"image":img,"option":"pencil"}#修改选项
    request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/style_trans"

    access_token = '24.e174e7b36d48d2d74590ffcb2a234eda.2592000.1647152243.282335-25595602'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        result = response.json()
        # print(type(result))
        print(result)
        image = result['image']
        imgData = base64.b64decode(image)
        f1 = open('D:\dataset\photo2sketch/testB/'+str(i)+'.png', 'wb')
        f1.write(imgData)

# wsi_mask_paths = glob.glob(os.path.join('*.png'))#得到当前目录下的所有图片
for i in range(1001,1214):
    transfer_picture(i)
    print(i)
