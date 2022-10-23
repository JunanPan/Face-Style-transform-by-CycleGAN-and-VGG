import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ft.v20200304 import ft_client, models
import base64
import time



for i in range(801,1001):
    time.sleep(0.8)
    f = open('D:\dataset\photo2cartoon1/testA/' + str(i) + '.png', 'rb')
    img = base64.b64encode(f.read())
    try:
        cred = credential.Credential("AKIDRSL4aswmOmwqVxA8qIVWFj283i3A2h4B", "1zqTsEQYdgZFi0EMm4HQRM7e09EJNJC3")
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ft.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = ft_client.FtClient(cred, "ap-shanghai", clientProfile)

        req = models.FaceCartoonPicRequest()
        params = {
        # 'Region': 'ap-beijing',
        'Action':'FaceCartoonPic',
        'Version':'2020-03-04',
        'Image':str(img, encoding='utf-8'),
        'RspImgType':'base64'
        }
        # print(str(img))
        req.from_json_string(json.dumps(params))
        print('1')
        resp = client.FaceCartoonPic(req)
        print('2')
        print(resp.to_json_string())
        resp = json.loads(resp.to_json_string())
        imgData = resp['ResultImage']
        imgData = base64.b64decode(imgData)
        f1 = open('D:\dataset\photo2cartoon1/testB/' + str(i) + '.png', 'wb')
        f1.write(imgData)
    except TencentCloudSDKException as err:
        print(err)