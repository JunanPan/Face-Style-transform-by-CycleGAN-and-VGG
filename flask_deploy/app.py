# coding:utf-8
#直接运行 python app.py 就行了，定义了端口号8987，在本机上访问 '127.0.0.1:8987/upload'
#tips： 如果是在其他机器上访问，把127.0.0.1的IP换成服务器的IP就行了
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from PIL import Image
from datetime import timedelta
import os
# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# 这个函数不用看
def concat2picture(pic1_name,pic2_name):
    def left_right_join(png1, png2):
        # img1, img2 = Image.open(png1), Image.open(png2)
        img1, img2 = png1, png2
        # # 统一图片尺寸，可以自定义设置（宽，高）
        # img1 = img1.resize((256, 1000), Image.ANTIALIAS)
        # img2 = img2.resize((1500, 1000), Image.ANTIALIAS)
        size1, size2 = img1.size, img2.size

        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save('./static/images/sumiao_cartoon_concat.png')
    sumiao_path = './static/images/'+pic1_name
    cartoon_path = './static/images/'+pic2_name
    #size=256#预处理，原图片大小 lxd
    sumiao_img = Image.open(sumiao_path)
    cartoon_img = Image.open(cartoon_path)
    #两个图片size一样
    size = sumiao_img.size #(364,369)
    sumiao_cropped = sumiao_img.crop((0, 0, size[0]/2,size[1]))#(左上，右下)
    sumiao_cropped.save("./static/images/sumiao_cropped.png")
    cartoon_cropped = cartoon_img.crop((size[0]/2,0,size[0],size[1]))
    cartoon_cropped.save("./static/images/cartoon_cropped.png")
    left_right_join(sumiao_cropped,cartoon_cropped)

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        # upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        upload_path = os.path.join(basepath, 'static/images','test.jpg')
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        # cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        #送入模型跑==========================================
        # os.system('python ../test.py --photo_path ./static/images/test.jpg --save_path ./static/images/res_sumiao.png --model sumiao')
        # os.system('python ../test.py --photo_path ./static/images/test.jpg --save_path ./static/images/res_cartoon.png --model cartoon')
        # concat2picture('res_sumiao.png','res_cartoon.png')
        #====================================================


        return render_template('upload_ok.html', userinput=user_input, val1=time.time())
        # return render_template('upload_ok.html', fileName=upload_path)

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=8987, debug=True)
