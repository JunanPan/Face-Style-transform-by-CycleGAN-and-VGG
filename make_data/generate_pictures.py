#使用预训练的styleGAN网络生成图片
import os
import pickle
import numpy as np
import PIL.Image
import generate_picture.dnnlib.tflib as tflib

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

def text_save(file, data):  # 保存生成代码
    for i in range(len(data[0])):
        s = str(data[0][i])+'\n'
        file.write(s)

def main():
    # 初始化tf
    tflib.init_tf()

    # 加载预训练模型
    model_path = 'model/generator_wanghong.pkl'  # 生成网红模型 1024*1024像素

    # 准备结果文件
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/generate_code', exist_ok=True)

    with open(model_path, "rb") as f:#加载预训练模型
        _G, _D, Gs = pickle.load(f, encoding='latin1')
        #合成网络（Synthesis network，也称：函数g）
    # 查看网络细节
    Gs.print_layers()

    # 生成图片
    generate_num = 2
    for i in range(generate_num):

        # 生成器输入 latents
        latents = np.random.randn(1, Gs.input_shape[1])

        #保存latents
        txt_filename = os.path.join(result_dir, 'generate_code/' + str(i).zfill(4) + '.txt')
        file = open(txt_filename, 'w')
        text_save(file, latents)

        # 生成图片
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)#定义输出图片的转换格式
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)#Gs.run()，输入和输出均为numpy数组

        # 保存图片
        png_filename = os.path.join(result_dir, str(i).zfill(4)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

        file.close()

if __name__ == "__main__":
    main()
